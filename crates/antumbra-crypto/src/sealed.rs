//! The sealed-file format (`ASF1`) for what antumbra writes outside its database.
//!
//! ```text
//! header  "ASF1" | version 1 | suite 1 | salt[32] | len | key ref | len | purpose
//! body    chunk 0 | chunk 1 | ... | last chunk
//! ```
//!
//! The body is the STREAM construction (Hoang, Reyhanitabar, Rogaway and Vizár) over
//! XChaCha20-Poly1305, from `aead-stream`: the plaintext in chunks of [`CHUNK`] bytes, each
//! sealed with a 16-byte tag under a nonce that carries the chunk's counter and, for the last
//! chunk alone, a flag. So a file whose chunks were changed, reordered, dropped, cut or extended
//! fails to open. The header is the associated data of every chunk, so it cannot be changed
//! either.
//!
//! The file's key and nonce come from the keyring key the header names, the file's random salt,
//! its key reference and its purpose (HKDF-SHA-256), so no two files share a key and a file
//! sealed for one purpose (one workspace's adapter, say) does not open as another.
//!
//! A [`SealWriter`] dropped without [`SealWriter::finish`] leaves no last chunk, so what it wrote
//! never opens.

use std::fs;
use std::io::{self, Read, Write};
use std::path::Path;

use aead_stream::{DecryptorBE32, EncryptorBE32, StreamBE32};
use chacha20poly1305::aead::Payload;
use chacha20poly1305::{KeyInit, XChaCha20Poly1305};
use zeroize::Zeroizing;

use crate::kdf::hkdf;
use crate::key::random;
use crate::keyring::KeyRef;
use crate::{Error, Key, Result};

/// The first four bytes of every sealed file.
pub const MAGIC: [u8; 4] = *b"ASF1";
const VERSION: u8 = 1;
/// XChaCha20-Poly1305 under STREAM (32-bit big-endian counter).
const SUITE_XCHACHA20POLY1305_STREAM: u8 = 1;
/// The plaintext size of every chunk but the last.
pub const CHUNK: usize = 64 * 1024;
const TAG: usize = 16;
const SALT_LEN: usize = 32;
/// The XChaCha20 nonce is 24 bytes; STREAM keeps the last 5 for its counter and flag.
const NONCE_PREFIX_LEN: usize = 19;
const LABEL: &[u8] = b"antumbra/sealed-file/v1";

type Nonce = aead_stream::Nonce<XChaCha20Poly1305, StreamBE32<XChaCha20Poly1305>>;

/// What a sealed file's header says, in the clear.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// The keyring key the file is sealed under.
    pub key_ref: KeyRef,
    /// What the file is for, chosen by the writer (`"adapter"`, `"export"`, ...), and checked
    /// when it is opened.
    pub purpose: String,
    salt: [u8; SALT_LEN],
}

impl Header {
    fn encode(&self) -> Result<Vec<u8>> {
        let key_ref = self.key_ref.to_string();
        let mut out = Vec::with_capacity(6 + SALT_LEN + 2 + key_ref.len() + self.purpose.len());
        out.extend_from_slice(&MAGIC);
        out.push(VERSION);
        out.push(SUITE_XCHACHA20POLY1305_STREAM);
        out.extend_from_slice(&self.salt);
        push_short(&mut out, key_ref.as_bytes(), "key reference")?;
        push_short(&mut out, self.purpose.as_bytes(), "purpose")?;
        Ok(out)
    }

    /// Reads a header, returning it and its exact bytes (the associated data of every chunk).
    fn read<R: Read>(reader: &mut R) -> Result<(Self, Vec<u8>)> {
        let mut fixed = [0u8; 6 + SALT_LEN];
        reader.read_exact(&mut fixed).map_err(short_header)?;
        if fixed[..4] != MAGIC {
            return Err(Error::format("sealed file", "it does not start with ASF1"));
        }
        if fixed[4] != VERSION {
            return Err(Error::format(
                "sealed file",
                format!("version {} is not one this build reads", fixed[4]),
            ));
        }
        if fixed[5] != SUITE_XCHACHA20POLY1305_STREAM {
            return Err(Error::format(
                "sealed file",
                format!("cipher suite {} is not one this build reads", fixed[5]),
            ));
        }
        let mut salt = [0u8; SALT_LEN];
        salt.copy_from_slice(&fixed[6..]);
        let mut raw = fixed.to_vec();
        let key_ref = read_short(reader, &mut raw)?;
        let purpose = read_short(reader, &mut raw)?;
        let key_ref = String::from_utf8(key_ref)
            .map_err(|_| Error::format("sealed file", "the key reference is not UTF-8"))?
            .parse()?;
        let purpose = String::from_utf8(purpose)
            .map_err(|_| Error::format("sealed file", "the purpose is not UTF-8"))?;
        Ok((
            Self {
                key_ref,
                purpose,
                salt,
            },
            raw,
        ))
    }

    /// The file's own cipher and STREAM nonce prefix.
    fn file_cipher(&self, keyring_key: &Key) -> Result<(XChaCha20Poly1305, Nonce)> {
        let key_ref = self.key_ref.to_string();
        let mut okm = Zeroizing::new([0u8; 32 + NONCE_PREFIX_LEN]);
        hkdf(
            keyring_key.expose(),
            &self.salt,
            &[
                LABEL,
                &[0],
                key_ref.as_bytes(),
                &[0],
                self.purpose.as_bytes(),
            ],
            okm.as_mut(),
        )?;
        let cipher = XChaCha20Poly1305::new_from_slice(&okm[..32])
            .map_err(|_| Error::Kdf("derived key has the wrong length".into()))?;
        let mut prefix = [0u8; NONCE_PREFIX_LEN];
        prefix.copy_from_slice(&okm[32..]);
        Ok((cipher, Nonce::from(prefix)))
    }
}

fn push_short(out: &mut Vec<u8>, field: &[u8], what: &str) -> Result<()> {
    let len = u8::try_from(field.len()).map_err(|_| {
        Error::format(
            "sealed file",
            format!("the {what} is longer than 255 bytes"),
        )
    })?;
    out.push(len);
    out.extend_from_slice(field);
    Ok(())
}

fn read_short<R: Read>(reader: &mut R, raw: &mut Vec<u8>) -> Result<Vec<u8>> {
    let mut len = [0u8; 1];
    reader.read_exact(&mut len).map_err(short_header)?;
    let mut field = vec![0u8; usize::from(len[0])];
    reader.read_exact(&mut field).map_err(short_header)?;
    raw.push(len[0]);
    raw.extend_from_slice(&field);
    Ok(field)
}

fn short_header(err: io::Error) -> Error {
    if err.kind() == io::ErrorKind::UnexpectedEof {
        Error::format("sealed file", "it ends inside its header")
    } else {
        Error::Io(err)
    }
}

/// Whether `bytes` begin like a sealed file. Says nothing about whether it opens.
pub fn is_sealed(bytes: &[u8]) -> bool {
    bytes.len() >= MAGIC.len() && bytes[..MAGIC.len()] == MAGIC
}

/// Writes a sealed file to `inner`, a chunk at a time. Call [`SealWriter::finish`]: a writer
/// dropped without it leaves a file that never opens.
pub struct SealWriter<W: Write> {
    inner: W,
    encryptor: Option<EncryptorBE32<XChaCha20Poly1305>>,
    aad: Vec<u8>,
    buffer: Zeroizing<Vec<u8>>,
}

impl<W: Write> SealWriter<W> {
    /// Starts a sealed file under the keyring key `key_ref` (whose bytes are `keyring_key`), for
    /// `purpose`, and writes its header.
    pub fn new(mut inner: W, keyring_key: &Key, key_ref: &KeyRef, purpose: &str) -> Result<Self> {
        let header = Header {
            key_ref: key_ref.clone(),
            purpose: purpose.to_owned(),
            salt: random()?,
        };
        let aad = header.encode()?;
        let (cipher, nonce) = header.file_cipher(keyring_key)?;
        inner.write_all(&aad)?;
        Ok(Self {
            inner,
            encryptor: Some(EncryptorBE32::from_aead(cipher, &nonce)),
            aad,
            buffer: Zeroizing::new(Vec::with_capacity(CHUNK)),
        })
    }

    fn emit_next(&mut self) -> Result<()> {
        let encryptor = self
            .encryptor
            .as_mut()
            .ok_or_else(|| Error::Refused("the sealed file is already finished".into()))?;
        let sealed = encryptor
            .encrypt_next(Payload {
                msg: &self.buffer,
                aad: &self.aad,
            })
            .map_err(|_| Error::Authentication)?;
        self.inner.write_all(&sealed)?;
        self.buffer.clear();
        Ok(())
    }

    /// Seals what is left as the last chunk, flushes, and returns the inner writer.
    pub fn finish(mut self) -> Result<W> {
        let encryptor = self
            .encryptor
            .take()
            .ok_or_else(|| Error::Refused("the sealed file is already finished".into()))?;
        let sealed = encryptor
            .encrypt_last(Payload {
                msg: &self.buffer,
                aad: &self.aad,
            })
            .map_err(|_| Error::Authentication)?;
        self.inner.write_all(&sealed)?;
        self.inner.flush()?;
        self.buffer.clear();
        Ok(self.inner)
    }
}

impl<W: Write> Write for SealWriter<W> {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let mut taken = 0;
        while taken < data.len() {
            // A full chunk is sealed only once more data arrives, so that the last chunk is
            // always the one `finish` seals.
            if self.buffer.len() == CHUNK {
                self.emit_next()?;
            }
            let room = CHUNK - self.buffer.len();
            let take = room.min(data.len() - taken);
            self.buffer.extend_from_slice(&data[taken..taken + take]);
            taken += take;
        }
        Ok(data.len())
    }

    /// Flushes the inner writer. A partial chunk stays buffered until more data or `finish`.
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Reads a sealed file, opening one chunk at a time.
pub struct OpenReader<R: Read> {
    inner: R,
    decryptor: Option<DecryptorBE32<XChaCha20Poly1305>>,
    aad: Vec<u8>,
    header: Header,
    plain: Zeroizing<Vec<u8>>,
    at: usize,
    carry: Option<u8>,
}

impl<R: Read> OpenReader<R> {
    /// Reads the header, checks the file was sealed for `expected_purpose`, and asks `key_for`
    /// for the keyring key it names.
    pub fn new<F>(mut inner: R, expected_purpose: &str, key_for: F) -> Result<Self>
    where
        F: FnOnce(&KeyRef) -> Result<Key>,
    {
        let (header, aad) = Header::read(&mut inner)?;
        if header.purpose != expected_purpose {
            return Err(Error::Purpose {
                expected: expected_purpose.to_owned(),
                found: header.purpose,
            });
        }
        let keyring_key = key_for(&header.key_ref)?;
        let (cipher, nonce) = header.file_cipher(&keyring_key)?;
        Ok(Self {
            inner,
            decryptor: Some(DecryptorBE32::from_aead(cipher, &nonce)),
            aad,
            header,
            plain: Zeroizing::new(Vec::new()),
            at: 0,
            carry: None,
        })
    }

    /// What the header says.
    pub fn header(&self) -> &Header {
        &self.header
    }

    /// Reads up to one chunk and its tag, plus one byte past it to tell whether it is the last.
    fn next_chunk(&mut self) -> Result<bool> {
        let want = CHUNK + TAG + 1;
        let mut chunk = Vec::with_capacity(want);
        if let Some(byte) = self.carry.take() {
            chunk.push(byte);
        }
        let mut block = [0u8; 8192];
        while chunk.len() < want {
            let ask = (want - chunk.len()).min(block.len());
            let got = self.inner.read(&mut block[..ask])?;
            if got == 0 {
                break;
            }
            chunk.extend_from_slice(&block[..got]);
        }
        let last = chunk.len() < want;
        if !last {
            self.carry = chunk.pop();
        }
        if chunk.len() < TAG {
            return Err(Error::Authentication);
        }
        let plain = if last {
            let decryptor = self.decryptor.take().ok_or(Error::Authentication)?;
            decryptor.decrypt_last(Payload {
                msg: &chunk,
                aad: &self.aad,
            })
        } else {
            let decryptor = self.decryptor.as_mut().ok_or(Error::Authentication)?;
            decryptor.decrypt_next(Payload {
                msg: &chunk,
                aad: &self.aad,
            })
        }
        .map_err(|_| Error::Authentication)?;
        self.plain = Zeroizing::new(plain);
        self.at = 0;
        Ok(last)
    }
}

impl<R: Read> Read for OpenReader<R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        while self.at == self.plain.len() {
            if self.decryptor.is_none() {
                return Ok(0);
            }
            self.next_chunk()?;
        }
        let n = out.len().min(self.plain.len() - self.at);
        out[..n].copy_from_slice(&self.plain[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

/// Seals `plaintext` whole.
pub fn seal(
    plaintext: &[u8],
    keyring_key: &Key,
    key_ref: &KeyRef,
    purpose: &str,
) -> Result<Vec<u8>> {
    let mut writer = SealWriter::new(Vec::new(), keyring_key, key_ref, purpose)?;
    writer.write_all(plaintext)?;
    writer.finish()
}

/// Opens a whole sealed file sealed for `expected_purpose`.
pub fn open<F>(sealed: &[u8], expected_purpose: &str, key_for: F) -> Result<Zeroizing<Vec<u8>>>
where
    F: FnOnce(&KeyRef) -> Result<Key>,
{
    let mut reader = OpenReader::new(sealed, expected_purpose, key_for)?;
    let mut plain = Zeroizing::new(Vec::new());
    reader.read_to_end(&mut plain)?;
    Ok(plain)
}

/// Seals `plaintext` into `path`, atomically: it is written beside the path and renamed over it,
/// so a crash leaves the old file or the new one, never half of one.
pub fn seal_to_path(
    path: &Path,
    plaintext: &[u8],
    keyring_key: &Key,
    key_ref: &KeyRef,
    purpose: &str,
) -> Result<()> {
    let sealed = seal(plaintext, keyring_key, key_ref, purpose)?;
    crate::keyring::write_atomically(path, &sealed)
}

/// Opens the sealed file at `path`.
pub fn open_path<F>(path: &Path, expected_purpose: &str, key_for: F) -> Result<Zeroizing<Vec<u8>>>
where
    F: FnOnce(&KeyRef) -> Result<Key>,
{
    let file = fs::File::open(path)?;
    let mut reader = OpenReader::new(io::BufReader::new(file), expected_purpose, key_for)?;
    let mut plain = Zeroizing::new(Vec::new());
    reader.read_to_end(&mut plain)?;
    Ok(plain)
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = std::result::Result<(), Box<dyn std::error::Error>>;

    fn fixture() -> Result<(Key, KeyRef)> {
        Ok((Key::from_slice(&[9; 32])?, "files/1".parse()?))
    }

    fn opener(key: &Key) -> impl FnOnce(&KeyRef) -> Result<Key> + '_ {
        move |_| Ok(key.clone())
    }

    fn pattern(len: usize) -> Vec<u8> {
        (0..len).map(|i| (i * 31 % 251) as u8).collect()
    }

    fn header_len(sealed: &[u8]) -> usize {
        let key_ref_len = usize::from(sealed[38]);
        let purpose_len = usize::from(sealed[39 + key_ref_len]);
        40 + key_ref_len + purpose_len
    }

    fn assert_refused(result: Result<Zeroizing<Vec<u8>>>) {
        assert!(
            matches!(result, Err(Error::Authentication)),
            "expected an authentication failure, got {:?}",
            result.map(|p| p.len())
        );
    }

    #[test]
    fn every_size_round_trips() -> TestResult {
        let (key, key_ref) = fixture()?;
        for len in [0, 1, CHUNK - 1, CHUNK, CHUNK + 1, 3 * CHUNK, 3 * CHUNK + 7] {
            let plain = pattern(len);
            let sealed = seal(&plain, &key, &key_ref, "adapter")?;
            assert!(is_sealed(&sealed));
            let chunks = len / CHUNK + usize::from(len % CHUNK != 0 || len == 0);
            assert_eq!(
                sealed.len(),
                header_len(&sealed) + len + chunks.max(1) * TAG,
                "len {len}"
            );
            let opened = open(&sealed, "adapter", opener(&key))?;
            assert_eq!(opened.as_slice(), plain.as_slice(), "len {len}");
        }
        Ok(())
    }

    #[test]
    fn small_writes_seal_what_one_write_seals() -> TestResult {
        let (key, key_ref) = fixture()?;
        let plain = pattern(2 * CHUNK + 100);
        let mut writer = SealWriter::new(Vec::new(), &key, &key_ref, "corpus")?;
        for piece in plain.chunks(777) {
            writer.write_all(piece)?;
        }
        let sealed = writer.finish()?;
        assert_eq!(
            open(&sealed, "corpus", opener(&key))?.as_slice(),
            plain.as_slice()
        );
        Ok(())
    }

    #[test]
    fn two_seals_of_the_same_bytes_differ() -> TestResult {
        let (key, key_ref) = fixture()?;
        let a = seal(b"same", &key, &key_ref, "export")?;
        let b = seal(b"same", &key, &key_ref, "export")?;
        assert_ne!(a, b, "each file has its own salt, so its own key");
        Ok(())
    }

    #[test]
    fn the_wrong_key_is_refused() -> TestResult {
        let (key, key_ref) = fixture()?;
        let sealed = seal(b"secret", &key, &key_ref, "export")?;
        let other = Key::from_slice(&[8; 32])?;
        assert_refused(open(&sealed, "export", opener(&other)));
        Ok(())
    }

    #[test]
    fn another_purpose_is_refused_before_any_key_is_asked_for() -> TestResult {
        let (key, key_ref) = fixture()?;
        let sealed = seal(b"weights", &key, &key_ref, "adapter/ws:a")?;
        let result = open(&sealed, "adapter/ws:b", |_| {
            Err(Error::Refused("no key should be asked for".into()))
        });
        assert!(matches!(result, Err(Error::Purpose { .. })));
        Ok(())
    }

    #[test]
    fn a_changed_byte_anywhere_is_refused() -> TestResult {
        let (key, key_ref) = fixture()?;
        let sealed = seal(&pattern(CHUNK + 500), &key, &key_ref, "export")?;
        let header = header_len(&sealed);
        // The salt, the first chunk's body and tag, and the last chunk.
        for at in [10, header + 5, header + CHUNK + TAG - 1, sealed.len() - 3] {
            let mut changed = sealed.clone();
            changed[at] ^= 0x01;
            assert_refused(open(&changed, "export", opener(&key)));
        }
        Ok(())
    }

    #[test]
    fn a_changed_header_is_refused() -> TestResult {
        let (key, key_ref) = fixture()?;
        let sealed = seal(b"x", &key, &key_ref, "export")?;
        let mut bad_magic = sealed.clone();
        bad_magic[0] = b'B';
        assert!(matches!(
            open(&bad_magic, "export", opener(&key)),
            Err(Error::Format { .. })
        ));
        let mut bad_version = sealed.clone();
        bad_version[4] = 2;
        assert!(matches!(
            open(&bad_version, "export", opener(&key)),
            Err(Error::Format { .. })
        ));
        Ok(())
    }

    #[test]
    fn truncation_at_any_point_is_refused() -> TestResult {
        let (key, key_ref) = fixture()?;
        let sealed = seal(&pattern(2 * CHUNK + 10), &key, &key_ref, "export")?;
        let header = header_len(&sealed);
        // Inside the last chunk, at a chunk boundary (the last chunk dropped), and right after
        // the header (no chunk at all).
        for keep in [
            sealed.len() - 1,
            header + 2 * (CHUNK + TAG),
            header + CHUNK + TAG,
            header,
        ] {
            assert_refused(open(&sealed[..keep], "export", opener(&key)));
        }
        let cut_header = &sealed[..header - 1];
        assert!(matches!(
            open(cut_header, "export", opener(&key)),
            Err(Error::Format { .. })
        ));
        Ok(())
    }

    #[test]
    fn appended_bytes_are_refused() -> TestResult {
        let (key, key_ref) = fixture()?;
        let sealed = seal(&pattern(CHUNK), &key, &key_ref, "export")?;
        let mut longer = sealed.clone();
        longer.extend_from_slice(&[0u8; TAG + 1]);
        assert_refused(open(&longer, "export", opener(&key)));
        Ok(())
    }

    #[test]
    fn reordered_or_dropped_chunks_are_refused() -> TestResult {
        let (key, key_ref) = fixture()?;
        let sealed = seal(&pattern(3 * CHUNK + 1), &key, &key_ref, "export")?;
        let header = header_len(&sealed);
        let full = CHUNK + TAG;
        let chunk = |i: usize| &sealed[header + i * full..header + (i + 1) * full];
        let mut swapped = sealed[..header].to_vec();
        swapped.extend_from_slice(chunk(1));
        swapped.extend_from_slice(chunk(0));
        swapped.extend_from_slice(&sealed[header + 2 * full..]);
        assert_refused(open(&swapped, "export", opener(&key)));
        let mut dropped = sealed[..header + full].to_vec();
        dropped.extend_from_slice(&sealed[header + 2 * full..]);
        assert_refused(open(&dropped, "export", opener(&key)));
        Ok(())
    }

    #[test]
    fn a_chunk_from_another_file_is_refused() -> TestResult {
        let (key, key_ref) = fixture()?;
        let a = seal(&pattern(2 * CHUNK), &key, &key_ref, "export")?;
        let b = seal(&pattern(2 * CHUNK), &key, &key_ref, "export")?;
        let header = header_len(&a);
        let full = CHUNK + TAG;
        let mut mixed = a[..header + full].to_vec();
        mixed.extend_from_slice(&b[header + full..]);
        assert_refused(open(&mixed, "export", opener(&key)));
        Ok(())
    }

    #[test]
    fn a_writer_dropped_without_finish_leaves_a_file_that_never_opens() -> TestResult {
        let (key, key_ref) = fixture()?;
        let mut out = Vec::new();
        {
            let mut writer = SealWriter::new(&mut out, &key, &key_ref, "export")?;
            writer.write_all(&pattern(CHUNK + 10))?;
            writer.flush()?;
        }
        assert_refused(open(&out, "export", opener(&key)));
        Ok(())
    }

    #[test]
    fn seal_to_path_and_open_path_round_trip() -> TestResult {
        let (key, key_ref) = fixture()?;
        let dir = std::env::temp_dir().join(format!(
            "antumbra-crypto-test-{}",
            data_encoding::HEXLOWER.encode(&random::<8>()?)
        ));
        fs::create_dir_all(&dir)?;
        let path = dir.join("adapter.safetensors");
        seal_to_path(&path, b"weights", &key, &key_ref, "adapter")?;
        assert_eq!(
            open_path(&path, "adapter", opener(&key))?.as_slice(),
            b"weights"
        );
        fs::remove_dir_all(&dir)?;
        Ok(())
    }
}
