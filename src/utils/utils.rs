use bincode::{ serialize, deserialize };
use lazy_static::lazy_static;
use openssl::ssl::{ SslMethod, SslAcceptor, SslFiletype, SslAcceptorBuilder };
use std::{ fs::File, io::{ BufReader, prelude::* } };

use crate::chains::chain_types::{ BlockChain, Block };

pub(crate) const NODE_IDENTIFIER: &'static str = "7d12866ef4a9404386df097afd43639f"; // any string you can place here
// easily to handle multiple error
pub(crate) type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync + 'static>>;


pub(crate) fn load_ssl() -> Result<SslAcceptorBuilder> {
    let mut builder = SslAcceptor::mozilla_intermediate(SslMethod::tls())?;
    builder.set_private_key_file("ssl_keys/server.pem", SslFiletype::PEM)?;
    builder.set_certificate_chain_file("ssl_keys/crt.pem")?;
    Ok(builder)
}


// read project config file
pub(crate) fn server_config() -> Result<toml::Value> {
    let config = File::open(concat!(env!("CARGO_MANIFEST_DIR"),"/yt-chain.toml"))?;
    let mut buff = BufReader::new(config);
    let mut contents = String::new();
    buff.read_to_string(&mut contents)?;
    
    let value = contents.parse::<toml::Value>()?;
    Ok(value["production"].clone())
}


fn concatenate_block(_key: &[u8], old_chains: Option<&[u8]>, new_block: &[u8]) -> Option<Vec<u8>> {
    let old_chains: bincode::Result<BlockChain> = deserialize(old_chains?);
    let new_block: bincode::Result<Block> = deserialize(new_block);
    
    match (old_chains.is_ok(), new_block.is_ok()) {
        (true, true) => {
            let mut chains = old_chains.unwrap();
            chains.chain.push(new_block.unwrap());
            serialize(&chains).ok() // convert to option
        }
        _ => None
    }
}

pub(crate) fn db_config() -> sled::Result<sled::Db> {
    let db_config = sled::ConfigBuilder::default()
                    // create a folder for store database file
                    .path(concat!(env!("CARGO_MANIFEST_DIR"), "/chain_db/"))
                    .async_io(true) 
                    .async_io_threads(12) // enable 12 threads
                    .cache_capacity(100_000_000) // size of databse file, 100M
                    .use_compression(true)
                    .flush_every_ms(Some(1000))
                    .snapshot_after_ops(100_000)
                    .merge_operator(concatenate_block)
                    .build();
                    
    sled::Db::start(db_config)
}