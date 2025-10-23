# source: https://github.com/mCodingLLC/VideosSampleCode/blob/master/videos/064_md5_a_broken_secure_hash_algorithm_python_implementation/md5.py

'''
NOTE: REFRESHERS
----------------
    Boolean Operations:
        &   →   AND
        |   →   OR
        ^   →   XOR (exclusive OR)
        ~   →   NOT (bitwise complement)
        <<  →   shift bits left; fill w/ 0s
        >>  →   shift bits right; fill w/ 0s

CONCEPTUAL FLOW
    Input → padding → chunking (64 B) → compression → final digest

COMPRESSION LOGIC
    Each of the 64 steps uses a nonlinear Boolean function, an additive constant, a message word, and a bit rotation.

DATAFLOW VISUALIZATION
    The four state registers `(A, B, C, D)` rotate each step.  
    After all 64 steps, they are _added back_ to the previous state to accumulate effects.
'''

from io import BytesIO
from typing import BinaryIO
import numpy as np


# predefined left-rotation amounts for each of the 64 MD5 steps
# each group of 16 corresponds to one "round" of MD5 operations
shift = [7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
         5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
         4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
         6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21]


# MD5 needs 64 constants (one per round)
# These constants introduce nonlinearity and fixed randomness
# constants derived from the sine function makes them "hard-to-predict" constants 
sines = np.abs(np.sin(np.arange(64) + 1))                       # absolute sine values for integers 1..64
sine_randomness = [int(x) for x in np.floor(2 ** 32 * sines)]   # multiplies each sine value (0–1) by 2³² → scales val to 32-bit range


# standard MD5 block and digest sizes (in bytes)
MD5_BLOCK_SIZE = 64     # 512 bits
MD5_DIGEST_SIZE = 16    # 128 bits


def left_rotate(x: int, y: int) -> int:
    '''
    Rotates the bits of a 32-bit integer `x` left by `y` positions.

        Moves bits to the left
        the bits that fall off left end
        wrap around to right end
        (keeps all bits; no data loss)

    Parameters:
    -----------
    x : int
        The 32-bit integer to rotate.
    y : int
        Number of bits to rotate by (0–31).

    Returns:
    --------
    int
        The left-rotated 32-bit integer.

    Example:
    --------
    >>> left_rotate(0b 1111 1111 0000 0000 1010 1010 1100 1100, 1)
                    0b 1111 1110 0000 0001 0101 0101 1001 1001
    '''

    # shift x left by y bits, bring the overflowed bits around to the right side
    # '& 0xffffffff' ensures the result fits in 32 bits (since Python ints are unbounded)
    #   “Take a 32-bit number, spin it left by y bits, and wrap anything that falls off back around to the right.”
    return ((x << (y & 31)) | ((x & 0xffffffff) >> (32 - (y & 31)))) & 0xffffffff


def bit_not(x: int) -> int:
    '''
    Bitwise NOT for a 32-bit integer.

    Parameters:
    -----------
    x : int
        Input integer to invert.

    Returns:
    --------
    int
        Bitwise complement (flipped bits) of the input.

    Example:
    --------
    >>> bit_not(0b11111111000000001010101011001100)
    0b00000000111111110101010100110011
    '''

    # since ~x in Python returns an infinite-precision integer,
    # we emulate 32-bit behavior by subtracting from 2^32-1
    return 4_294_967_295 - x # equivalent to (~x) & 0xffffffff


'''
---------------------------------------------------------------------
Logical "mixing" functions F, G, H, I
   Each takes three 32-bit inputs and returns one 32-bit output.
   They introduce nonlinear combinations of bits to achieve diffusion.
   (diffusion - ensure that changing one input bit changes many output bits; avalanche effect)
---------------------------------------------------------------------
'''
def F(b: int, c: int, d: int) -> int:
    # (b AND c) OR (NOT b AND d)
    # selects bits from c if b = 1, otherwise from d
    return (b & c) | (bit_not(b) & d)

def G(b: int, c: int, d: int) -> int:
    # (b AND d) OR (c AND NOT d)
    # similar logic, but slightly shifted bit dependencies
    return (b & d) | (c & bit_not(d))

def H(b: int, c: int, d: int) -> int:
    # XOR of all three inputs
    # flips bits if an odd number of inputs have that bit set
    return b ^ c ^ d

def I(b: int, c: int, d: int) -> int:
    # c XOR (b OR NOT d)
    # last nonlinear mixing function, introducing high diffusion
    return c ^ (b | bit_not(d))


# map each of the 64 steps to one of the four mixing functions
mixer_for_step = (
    [F for _ in range(16)] + 
    [G for _ in range(16)] + 
    [H for _ in range(16)] + 
    [I for _ in range(16)]
)


# message-word order (which 32-bit word to use each step)
# each round shuffles indices differently to increase diffusion
round_1_perm = [i for i in range(16)]                   # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
round_2_perm = [(5 * i + 1) % 16 for i in range(16)]    # [1, 6, 11, 0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12]
round_3_perm = [(3 * i + 5) % 16 for i in range(16)]    # [5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2]
round_4_perm = [(7 * i) % 16 for i in range(16)]        # [0, 7, 14, 5, 12, 3, 10, 1, 8, 15, 6, 13, 4, 11, 2, 9]

msg_idx_for_step = round_1_perm + round_2_perm + round_3_perm + round_4_perm



class MD5State:
    '''
    Internal mutable MD5 hashing state.

    Attributes:
    -----------
    length : int
        Total number of bytes processed so far.

    state : tuple[int, int, int, int]
        The four 32-bit words representing the MD5 internal hash state (A, B, C, D).

    n_filled_bytes : int
        Number of bytes currently stored in the internal 64-byte buffer.

    buffer : bytearray
        Working 64-byte buffer for message chunk accumulation.
    '''

    def __init__(self):
        '''
        Initializes a new MD5 hashing state with the standard initial vector.
        '''

        # initialize running byte count
        self.length = 0

        # initial 128-bit state values per RFC 1321 (little-endian constants)
        self.state = (0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476)

        # internal 64-byte working buffer
        self.n_filled_bytes = 0
        self.buffer = bytearray(MD5_BLOCK_SIZE)


    def digest(self) -> bytes:
        '''
        Returns the final digest (A, B, C, D) as a 16-byte little-endian byte string.

        Returns:
        --------
        bytes
            16-byte MD5 digest.
        '''

        # concatenate the four state words as 16 little-endian bytes
        return b''.join(x.to_bytes(length = 4, byteorder = 'little') for x in self.state)


    def hex_digest(self) -> str:
        '''
        Returns the digest as a 32-character hexadecimal string.

        Returns:
        --------
        str
            Hexadecimal MD5 digest.
        '''

        # convert final digest bytes to human-readable hexadecimal
        return self.digest().hex()


    def process(self, stream: BinaryIO) -> None:
        '''
        Reads data from a stream in 64-byte chunks and compresses each chunk.

        Parameters:
        -----------
        stream : BinaryIO
            Input byte stream to read and hash.
        '''
        assert self.n_filled_bytes < len(self.buffer)

        view = memoryview(self.buffer) # allows direct byte writing without copy

        # repeatedly read data from the stream until it's exhausted
        while bytes_read := stream.read(MD5_BLOCK_SIZE - self.n_filled_bytes):
            # fill part of the internal buffer with newly read data
            view[self.n_filled_bytes : self.n_filled_bytes + len(bytes_read)] = bytes_read

            # if we filled the buffer exactly (64 bytes), run the compression function
            if self.n_filled_bytes == 0 and len(bytes_read) == MD5_BLOCK_SIZE:
                self.compress(self.buffer)
                self.length += MD5_BLOCK_SIZE
            else:
                # otherwise accumulate bytes
                self.n_filled_bytes += len(bytes_read)
                # when buffer finally reaches 64 bytes, compress and reset it
                if self.n_filled_bytes == MD5_BLOCK_SIZE:
                    self.compress(self.buffer)
                    self.length += MD5_BLOCK_SIZE
                    self.n_filled_bytes = 0


    def finalize(self) -> None:
        '''
        Adds MD5 padding and message length, then processes the final block.

            Appends a single '1' bit (0x80), followed by '0' bits 
            until last 8 bytes hold the message length in bits (little-endian)
        '''
        assert self.n_filled_bytes < MD5_BLOCK_SIZE

        # count total number of bytes processed (including partial buffer)
        self.length += self.n_filled_bytes

        # append 1 bit (0x80) per spec; remainder zeros will come later
        self.buffer[self.n_filled_bytes] = 0b10000000
        self.n_filled_bytes += 1

        n_bytes_needed_for_len = 8 # space for 64-bit message length field

        # if not enough room in this block for both padding and length, flush now
        if self.n_filled_bytes + n_bytes_needed_for_len > MD5_BLOCK_SIZE:
            self.buffer[self.n_filled_bytes : ] = bytes(MD5_BLOCK_SIZE - self.n_filled_bytes)
            self.compress(self.buffer)
            self.n_filled_bytes = 0

        # zero-pad remainder of the buffer (up to last 8 bytes)
        self.buffer[self.n_filled_bytes : ] = bytes(MD5_BLOCK_SIZE - self.n_filled_bytes)

        # append total message length in *bits*, little-endian 64-bit integer
        bit_len_64 = (self.length * 8) % (2 ** 64)
        self.buffer[-n_bytes_needed_for_len : ] = bit_len_64.to_bytes(
            length = n_bytes_needed_for_len, 
            byteorder = 'little'
        )

        # compress the final padded block
        self.compress(self.buffer)


    def compress(self, msg_chunk: bytearray) -> None:
        '''
        Core MD5 compression function.

        Takes a 512-bit block (64-byte) and mixes it into current hash state.

        Parameters:
        -----------
        msg_chunk : bytearray
            64-byte message block to compress.
        '''
        assert len(msg_chunk) == MD5_BLOCK_SIZE  # 64 bytes, 512 bits

        # split 64-byte chunk into sixteen 32-bit little-endian integers
        msg_ints = [int.from_bytes(msg_chunk[i : i + 4], byteorder = 'little') 
                    for i in range(0, MD5_BLOCK_SIZE, 4)]
        assert len(msg_ints) == 16

        # unpack current state (A, B, C, D)
        a, b, c, d = self.state

        # perform 64 rounds of mixing
        for i in range(MD5_BLOCK_SIZE):     # MD5_BLOCK_SIZE == 64
            bit_mixer = mixer_for_step[i]   # choose F, G, H, or I
            msg_idx = msg_idx_for_step[i]   # select message word

            # -- MAIN OPERATION --
            # combine previous state with message + constant
            a = (a + bit_mixer(b, c, d) + msg_ints[msg_idx] + sine_randomness[i]) % (2 ** 32)
            # rotate bits left by shift[i]
            a = left_rotate(a, shift[i])
            # add to next state variable
            a = (a + b) % (2 ** 32)
            # cyclically rotate the state variables (a → d, b → a, etc)
            a, b, c, d = d, a, b, c

        # update global state by adding back into it (mod 2^32)
        self.state = (
            (self.state[0] + a) % (2 ** 32),
            (self.state[1] + b) % (2 ** 32),
            (self.state[2] + c) % (2 ** 32),
            (self.state[3] + d) % (2 ** 32),
        )

def md5(s: bytes) -> bytes:
    '''
    Compute the MD5 digest of a given byteS object.

    Parameters:
    -----------
    s : bytes
        Input message as bytes.

    Returns:
    --------
    bytes
        16-byte MD5 digest of the input.

    Example:
    --------
    >>> md5(b'abc').hex()
    '900150983cd24fb0d6963f7d28e17f72'
    '''
    state = MD5State()          # initialize hashing context
    state.process(BytesIO(s))   # feed the data through process()
    state.finalize()            # apply padding and finalize
    return state.digest()       # return raw 16-byte digest


def md5_file(file: BinaryIO) -> bytes:
    '''
    Convenience function to hash an open binary file stream.
    '''
    state = MD5State()
    state.process(file)
    state.finalize()
    return state.digest()


if __name__ == '__main__':
    import hashlib

    text = b'is mayonnaise an instrument'

    text_hashlib = hashlib.md5(text).hexdigest()
    text_scratch = md5(text).hex()

    print('text: ', text)
    print(f'hashlib: {text_hashlib}')
    print(f'scratch: {text_scratch}')
    print(text_hashlib == text_scratch)