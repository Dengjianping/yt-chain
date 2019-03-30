use bincode::{ serialize, deserialize };

use crate::chains::chain_types::{ BlockChain, Block };
use crate::utils::utils;


pub(crate) struct DBPool {
    pub(crate) conn: sled::Db,
}

pub(crate) struct DbState {
    pub(crate) db: actix::Addr<DBPool>,
}

type Key = String;
#[derive(Debug, Clone)]
pub(crate) enum DbOperation {
    Clear,
    Delete(Key),
    QueryByKey(Key),
    UpdateBlock(Key, Block),
    UpdateBlockChain(Key, BlockChain),
}

#[derive(Debug, Clone)]
pub(crate) enum ReturnType {
    BOOL(bool),
    BLOCKCHAIN(BlockChain),
    BLOCK(Block),
    NOTHING,
}


impl actix::Message for DbOperation {
    type Result = utils::Result<ReturnType>;
}

impl actix::Actor for DBPool {
    type Context = actix::SyncContext<Self>;
}


impl actix::Handler<DbOperation> for DBPool {
    type Result = utils::Result<ReturnType>;

    fn handle(&mut self, msg: DbOperation, _: &mut Self::Context) -> Self::Result {
        let conn: &sled::Db = &self.conn;
        
        let data: ReturnType = match msg {
            DbOperation::QueryByKey(key) => {
                let u8_key = key.as_bytes();
                match conn.contains_key(u8_key) {
                    Ok(has_key) => {
                        if has_key {
                            let chains = conn.get(u8_key)?.unwrap(); // need remove unwrap
                            
                            let chains: BlockChain = deserialize(&chains)?;
                            ReturnType::BLOCKCHAIN(chains)
                        } else {
                            // insert a key-value
                            let inited_block = BlockChain::new();
                            // serialize the new block
                            let y = serialize(&inited_block)?;
                            let _ = conn.set(u8_key, y);
                            ReturnType::BLOCKCHAIN(inited_block)
                        }
                    }
                    _ => ReturnType::NOTHING
                }
                
            }
            DbOperation::Delete(key) => {
                let u8_key = key.as_bytes();
                let deleted = conn.del(u8_key);
                ReturnType::BOOL(deleted.is_ok())
            }
            DbOperation::Clear => {
                let cleared = conn.clear();
                ReturnType::BOOL(cleared.is_ok())
            }
            DbOperation::UpdateBlock(key, block) => {
                let u8_key = key.as_bytes();
                
                match conn.contains_key(u8_key) {
                    Ok(has_key) => {
                        if has_key {
                            let block = serialize(&block)?;
                            let result = conn.merge(u8_key, block);
                            ReturnType::BOOL(result.is_ok())
                        } else {
                            ReturnType::BOOL(false)
                        }
                    }
                    _ => ReturnType::BOOL(false)
                }
            }
            DbOperation::UpdateBlockChain(key, block_chain) => {
                let u8_key = key.as_bytes();
                let serialized_block_chain = serialize(&block_chain)?;
                let result = conn.set(u8_key, serialized_block_chain);
                ReturnType::BOOL(result.is_ok())
            }
        };
        Ok(data)
    }
}