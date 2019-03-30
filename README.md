# yt-chain
[![Build Status](https://travis-ci.com/Dengjianping/minimal_block_chain.svg?branch=master)](https://travis-ci.com/Dengjianping/minimal_block_chain)

yt-chain is just a toy of blockchain inspired by this article titled [Learn Blockchains by Building One](https://hackernoon.com/learn-blockchains-by-building-one-117428612f46?gi=8e1bb887685f). Originally, that's python based blockchain, so I want to use rust to implement a one, and giving some improvements to the rust based.


### Stacks
- [rust](https://www.rust-lang.org/) 
- [actix-web](https://github.com/actix/actix-web). A blazingly fast **asynchronous** web framework based on Rust.
- [sled](https://github.com/spacejam/sled). This is a lock-free embedded database, purely written by rust, having high performance and easy usage, etc.
- [cuda](https://developer.nvidia.com/cuda-downloads). Use gpu to accelerate mining block instead of cpu.


### Requirements
1. Linux platform.
2. Latest stable/nightly rust.
3. CUDA installed(at least 8.0, compute capability at least 3.5).
   - This is for accelerating mining, I implemented some encryption algorithms like sha256/384/512.
4. Cmake(at least 3.8).
5. Gcc.


### Build && Run
1. Configure the yt-chain.toml file
- If you want cpu to mine the block, use following command the project

```
cargo run --release
```

- But if you want to try CUDA to mine the block, use this command to feel the feeling of flying

```
cargo run --features "cuda" --release
```

### Usage

Free to feel the blockchain.

- Mining.
- Get the whole blockchain.
- Transaction.
- Consensus.