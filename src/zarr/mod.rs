use derive_more::Debug;
use serde::{Deserialize, Serialize};
use std::ops::Index;

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrDataType {
    #[serde(rename = "|u1")]
    U1,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrVersion {
    #[serde(rename = "2")]
    V2 = 2,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrOrder {
    #[serde(rename = "C")]
    ColumnMajor,
    #[serde(rename = "F")]
    RowMajor,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrCompressionName {
    #[serde(rename = "lz4")]
    Lz4,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum ZarrCompressorId {
    #[serde(rename = "blosc")]
    Blosc,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrCompressor {
    blocksize: u8,
    clevel: u8,
    #[serde(rename = "cname")]
    compression_name: ZarrCompressionName,
    id: String,
    shuffle: u8,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrFilters {}

/*
{
    "chunks": [
        500,
        500,
        500
    ],
    "compressor": {
        "blocksize": 0,
        "clevel": 5,
        "cname": "lz4",
        "id": "blosc",
        "shuffle": 1
    },
    "dtype": "|u1",
    "fill_value": 0,
    "filters": null,
    "order": "C",
    "shape": [
        4251,
        3145,
        3432
    ],
    "zarr_format": 2
}%

*/

#[derive(Serialize, Deserialize, Debug, Clone)]
struct ZarrArrayDef {
    chunks: Vec<usize>,
    compressor: ZarrCompressor,
    dtype: String,
    fill_value: u8,
    filters: Option<ZarrFilters>,
    order: ZarrOrder,
    shape: Vec<usize>,
    zarr_format: u8,
}

pub struct ZarrArray<const N: usize, T> {
    path: String,
    def: ZarrArrayDef,
    phantom_t: std::marker::PhantomData<T>,
}

#[derive(Debug, Clone)]
struct BloscHeader {
    version: u8,
    version_lz: u8,
    flags: u8,
    typesize: usize,
    nbytes: usize,
    blocksize: usize,
    cbytes: usize,
}
impl BloscHeader {
    fn from_bytes(bytes: &[u8]) -> Self {
        BloscHeader {
            version: bytes[0],
            version_lz: bytes[1],
            flags: bytes[2],
            typesize: bytes[3] as usize,
            nbytes: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]) as usize,
            blocksize: u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
            cbytes: u32::from_le_bytes([bytes[12], bytes[13], bytes[14], bytes[15]]) as usize,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BloscChunk<T> {
    header: BloscHeader,
    offsets: Vec<u32>,
    #[debug(skip)]
    data: Vec<u8>,
    phantom_t: std::marker::PhantomData<T>,
}

impl BloscChunk<u8> {
    fn get(&self, index: usize) -> u8 {
        let block_idx = index * self.header.typesize as usize / self.header.blocksize as usize;
        let idx = (index * self.header.typesize as usize) % self.header.blocksize as usize;
        let block_offset = self.offsets[block_idx] as usize;
        let block_compressed_length =
            u32::from_le_bytes(self.data[block_offset..block_offset + 4].try_into().unwrap()) as usize;
        let block_compressed_data = &self.data[block_offset + 4..block_offset + block_compressed_length + 4];

        dbg!(
            "Block: {:?} {:?} {:x} {}",
            index,
            idx,
            block_idx,
            block_offset,
            block_compressed_length
        );

        let uncompressed = lz4_compression::decompress::decompress(&block_compressed_data).unwrap();

        uncompressed[idx]
    }
}

impl<const N: usize, T> ZarrArray<N, T> {
    fn load_chunk(&self, chunk_no: [usize; N]) -> BloscChunk<T> {
        let chunk_path = self.chunk_path(chunk_no);
        let chunk = std::fs::read(&chunk_path).unwrap();

        // parse 16 byte blosc header
        let header = BloscHeader::from_bytes(&chunk[0..16]);
        let mut offsets = vec![];
        for i in 0..((header.nbytes + header.blocksize - 1) / header.blocksize) as usize {
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
            phantom_t: std::marker::PhantomData,
        }
    }

    fn chunk_path(&self, chunk_no: [usize; N]) -> String {
        format!(
            "{}/{}",
            self.path,
            chunk_no.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(".")
        )
    }
}

impl<const N: usize> ZarrArray<N, u8> {
    pub fn from_path(path: &str) -> Self {
        // read and parse path/.zarray into ZarrArrayDef

        let zarray = std::fs::read_to_string(format!("{}/.zarray", path)).unwrap();
        println!("Read ZarrArrayDef: {}", zarray);
        let zarray_def = serde_json::from_str::<ZarrArrayDef>(&zarray).unwrap();

        println!("Loaded ZarrArrayDef: {:?}", zarray_def);

        assert!(zarray_def.shape.len() == N);

        ZarrArray {
            path: path.to_string(),
            def: zarray_def,
            phantom_t: std::marker::PhantomData,
        }
    }
    fn get(&self, index: [usize; N]) -> u8 {
        let chunk_no = index
            .iter()
            .zip(self.def.chunks.iter())
            .map(|(i, c)| i / c)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let chunk_offset = index
            .iter()
            .zip(self.def.chunks.iter())
            .map(|(i, c)| i % c)
            .collect::<Vec<_>>();

        let chunk = self.load_chunk(chunk_no);

        println!("Chunk: {:?}", chunk);
        let idx = chunk_offset
            .iter()
            .zip(self.def.chunks.iter())
            //.rev() // FIXME: only if row-major
            .fold(0, |acc, (i, c)| acc * c + i);
        println!("Index for {:?}: {:?}", chunk_offset, idx);
        chunk.get(idx)
    }
}

#[test]
pub fn test_zarr() {
    let zarr: ZarrArray<3, u8> =
        ZarrArray::from_path("/home/johannes/tmp/pap/fiber-predictions/7000_11249_predictions.zarr");

    let start_time = std::time::Instant::now();
    let comp = std::fs::read("/home/johannes/tmp/pap/fiber-predictions/7000_11249_predictions.zarr/b0.lz4").unwrap();
    println!("Read compressed file in {:?}", start_time.elapsed());
    let uncomp = lz4_compression::decompress::decompress(&comp).unwrap();

    std::fs::write(
        "/home/johannes/tmp/pap/fiber-predictions/7000_11249_predictions.zarr/b0.lz4.decomp",
        &uncomp,
    )
    .unwrap();
    println!("Decompressed to len {:?}", uncomp.len());

    let at = [1, 21, 118];
    let val = zarr.get(at);
    println!("Value at {:?}: {:?}", at, val);

    todo!()
}

/*
00000000  02 01 21 01 40 59 73 07  00 00 02 00 b4 02 69 00  |..!.@Ys.......i.|
00000010  93 12 00 00 f8 0e 00 00  a3 14 00 00 2a 38 00 00  |............*8..|
00000020  fd 24 00 00 ed 55 00 00  8a 71 00 00 82 b5 00 00  |.$...U...q......|
00000030  4c 87 00 00 49 9e 00 00  ef e1 00 00 79 cd 00 00  |L...I.......y...|
00000040  d2 fc 00 00 27 19 01 00  8c 48 01 00 77 32 01 00  |....'....H..w2..|
00000050  88 66 01 00 1b 85 01 00  e4 b1 01 00 4d 9c 01 00  |.f..........M...|
00000060  3e cf 01 00 9f eb 01 00  31 02 02 00 b8 20 02 00  |>.......1.... ..|
00000070  39 51 02 00 18 38 02 00  77 71 02 00 89 92 02 00  |9Q...8..wq......|
00000080  73 ca 02 00 44 ad 02 00  d5 ea 02 00 bb 07 03 00  |s...D...........|
00000090  ee 28 03 00 c2 41 03 00  43 63 03 00 e3 93 03 00  |.(...A..Cc......|
000000a0  f4 7b 03 00 b7 d1 03 00  d7 b6 03 00 69 f4 03 00  |.{..........i...|
000000b0  3c 12 04 00 8c 2e 04 00  78 4a 04 00 fb 67 04 00  |<.......xJ...g..|

|-0-|-1-|-2-|-3-|-4-|-5-|-6-|-7-|-8-|-9-|-A-|-B-|-C-|-D-|-E-|-F-|
  ^   ^   ^   ^ |     nbytes    |   blocksize   |    cbytes     |
  |   |   |   |
  |   |   |   +--typesize
  |   |   +------flags
  |   +----------versionlz
  +--------------version

02 version 2
01 version lz 1
21 flags = byte shuffle 0x01, compressor 0x20 >> 5 = 0x01 = lz4
01 typesize = 1 byte
40 59 73 07 nbytes = 125000000 = 500 * 500 * 500
00 00 02 00 blocksize = 0x20000 = 131072
b4 02 69 00 cbytes = 0x6902b4 = 6881972

93 12 00 00 f8 0e 00 00  a3 14 00 00 2a 38 00 00


*/
