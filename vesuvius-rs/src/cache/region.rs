//! Anonymous, lazily committed memory region backing one shard of decoded
//! chunks.
//!
//! The whole shard is reserved as virtual address space up front
//! (`MAP_NORESERVE`, so no commit charge); pages materialize on first write
//! and go back to the kernel with `madvise(MADV_DONTNEED)`, after which they
//! read as zeros. Untouched pages read as zeros too, which is what readers of
//! not-yet-filled slots expect.

use std::ops::Deref;

pub struct Region {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: the region is plain memory; concurrent access to a slot is ordered
// by the slot store's state bytes (writers hold the slot lock, readers
// Acquire-load RESIDENT before reading).
unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    pub fn reserve(len: usize) -> Self {
        // SAFETY: anonymous private mapping, no fd.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE,
                -1,
                0,
            )
        };
        assert!(
            ptr != libc::MAP_FAILED,
            "reserving {} GiB of address space failed: {}",
            len >> 30,
            std::io::Error::last_os_error()
        );
        Self { ptr: ptr.cast(), len }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Copy `bytes` to `off`.
    ///
    /// # Safety
    /// The caller must own the target range (the slot lock), so no other
    /// writer touches it and no reader treats it as published.
    pub unsafe fn write(&self, off: usize, bytes: &[u8]) {
        assert!(off + bytes.len() <= self.len);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(off), bytes.len());
    }

    /// Give the pages of `off..off + len` back to the kernel; they read as
    /// zeros afterwards. Same ownership contract as [`write`](Self::write).
    pub unsafe fn release(&self, off: usize, len: usize) {
        assert!(off + len <= self.len);
        assert_eq!(off % page_size(), 0, "release must be page aligned");
        let rc = libc::madvise(self.ptr.add(off).cast(), len, libc::MADV_DONTNEED);
        assert_eq!(rc, 0, "madvise(DONTNEED) failed: {}", std::io::Error::last_os_error());
    }
}

impl std::fmt::Debug for Region {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Region({:p}, {} bytes)", self.ptr, self.len)
    }
}

impl Deref for Region {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        // SAFETY: the mapping is `len` bytes and lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        // SAFETY: unmapping exactly what `reserve` mapped.
        unsafe { libc::munmap(self.ptr.cast(), self.len) };
    }
}

fn page_size() -> usize {
    // SAFETY: plain sysconf.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_read_release() {
        let r = Region::reserve(64 << 30);
        assert_eq!(r[12345], 0);
        unsafe { r.write(1 << 20, &[1, 2, 3]) };
        assert_eq!(&r[(1 << 20)..(1 << 20) + 3], &[1, 2, 3]);
        unsafe { r.release(1 << 20, 4096) };
        assert_eq!(&r[(1 << 20)..(1 << 20) + 3], &[0, 0, 0]);
    }
}
