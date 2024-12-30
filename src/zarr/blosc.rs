use crate::zstd_decompress;
use derive_more::Debug;
use memmap::MmapOptions;
use std::{collections::HashMap, fs::File};

#[derive(Debug, Clone)]
pub enum BloscShuffle {
    None,
    Bit,
    Byte,
}

#[derive(Debug, Clone)]
pub enum BloscCompressor {
    Blosclz,
    Lz4,
    Snappy,
    Zlib,
    Zstd,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BloscHeader {
    pub version: u8,
    pub version_lz: u8,
    pub flags: u8,
    pub typesize: usize,
    pub nbytes: usize,
    pub blocksize: usize,
    pub cbytes: usize,
    pub shuffle: BloscShuffle,
    pub compressor: BloscCompressor,
}
impl BloscHeader {
    fn from_bytes(bytes: &[u8]) -> Self {
        let flags = bytes[2];
        let shuffle = match flags & 0x7 {
            0 | 1 => BloscShuffle::None,
            2 => BloscShuffle::Byte,
            4 => BloscShuffle::Bit,
            x => panic!("Invalid shuffle value {x}"),
        };
        let compressor = match flags >> 5 {
            0 => BloscCompressor::Blosclz,
            1 => BloscCompressor::Lz4,
            2 => BloscCompressor::Snappy,
            3 => BloscCompressor::Zlib,
            4 => BloscCompressor::Zstd,
            x => panic!("Invalid compressor value {x}"),
        };

        BloscHeader {
            version: bytes[0],
            version_lz: bytes[1],
            flags,
            typesize: bytes[3] as usize,
            nbytes: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
            blocksize: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            cbytes: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize,
            shuffle,
            compressor,
        }
    }

    fn num_blocks(&self) -> usize {
        if self.blocksize == 0 {
            1
        } else {
            let res = (self.nbytes + self.blocksize - 1) / self.blocksize;
            res
        }
    }
}

#[derive(Debug)]
pub struct BloscChunk<T> {
    pub header: BloscHeader,
    offsets: Vec<u32>,
    #[debug(skip)]
    data: memmap::Mmap,
    file_name: String,
    phantom_t: std::marker::PhantomData<T>,
}

#[allow(dead_code)]
struct BloscBlock {
    id: u16, // FIXME
    data: Vec<u8>,
}

pub struct BloscContext {
    pub(crate) chunk: BloscChunk<u8>,
    cache: HashMap<usize, BloscBlock>,
    last_block_idx: usize,
    last_entry: Option<BloscBlock>,
}
impl BloscContext {
    pub fn get(&mut self, index: usize) -> u8 {
        let block_idx = index * self.chunk.header.typesize as usize / self.chunk.header.blocksize as usize;
        let idx = (index * self.chunk.header.typesize as usize) % self.chunk.header.blocksize as usize;

        /* println!(
            "get {} from {}, block_idx: {}, idx: {}",
            index, self.chunk.file_name, block_idx, idx
        ); */

        if block_idx == self.last_block_idx {
            self.last_entry.as_ref().unwrap().data[idx]
        } else if self.cache.contains_key(&block_idx) {
            let block = self.cache.remove(&block_idx).unwrap();
            if let Some(last_block) = self.last_entry.take() {
                self.cache.insert(self.last_block_idx, last_block);
            }
            let res = block.data[idx];
            self.last_block_idx = block_idx;
            self.last_entry = Some(block);
            res
        } else {
            if self.cache.len() > 1000 {
                self.cache.clear();
            }

            let uncompressed = self.chunk.decompress(block_idx);
            let res = uncompressed[idx];
            let block = BloscBlock {
                id: block_idx as u16,
                data: uncompressed,
            };
            self.cache.insert(block_idx, block);

            res
        }
    }
}

impl BloscChunk<u8> {
    pub fn load(filename: &str) -> Self {
        let file = File::open(filename).unwrap();
        let chunk = unsafe { MmapOptions::new().map(&file) }.unwrap();

        // parse 16 byte blosc header
        let header = BloscHeader::from_bytes(&chunk[0..16]);
        let num_blocks = header.num_blocks();
        let mut offsets = vec![];
        for i in 0..num_blocks as usize {
            offsets.push(u32::from_le_bytes([
                chunk[16 + i * 4],
                chunk[16 + i * 4 + 1],
                chunk[16 + i * 4 + 2],
                chunk[16 + i * 4 + 3],
            ]));
        }

        BloscChunk {
            header,
            offsets,
            data: chunk,
            file_name: filename.to_string(),
            phantom_t: std::marker::PhantomData,
        }
    }
    pub fn into_ctx(self) -> BloscContext {
        BloscContext {
            chunk: self,
            cache: HashMap::new(),
            last_block_idx: usize::MAX,
            last_entry: None,
        }
    }
    #[allow(dead_code)]
    pub fn get(&self, index: usize) -> u8 {
        let block_idx = index * self.header.typesize as usize / self.header.blocksize as usize;
        let idx = (index * self.header.typesize as usize) % self.header.blocksize as usize;

        self.decompress(block_idx)[idx]
    }
    fn decompress(&self, block_idx: usize) -> Vec<u8> {
        /* if block_idx >= self.num_blocks {
            panic!("Block index out of bounds for block {}", &self.file_name);
        } */
        let block_offset = self.offsets[block_idx] as usize;
        if block_offset + 4 >= self.data.len() {
            panic!("Block offset out of bounds for block {}", &self.file_name);
        }
        let block_compressed_length =
            u32::from_le_bytes(self.data[block_offset..block_offset + 4].try_into().unwrap()) as usize;
        let block_compressed_data = &self.data[block_offset + 4..block_offset + block_compressed_length + 4];

        match self.header.compressor {
            BloscCompressor::Lz4 => lz4_compression::decompress::decompress(&block_compressed_data)
                .expect(format!("Failed to decompress block {} in file {}", block_idx, self.file_name).as_str()),
            BloscCompressor::Zstd => zstd_decompress(block_compressed_data),
            _ => panic!(
                "Unsupported compressor: {:?} in file {:?}",
                self.header.compressor, self.file_name
            ),
        }
    }
}
