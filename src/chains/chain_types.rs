use bincode::{ serialize, deserialize };
use juniper::{ FieldResult, GraphQLEnum, GraphQLObject, GraphQLInputObject, graphql_object };
use serde_derive::{ Deserialize, Serialize };
use std::{ time::{ self, SystemTime }, collections::HashSet };
use sha2::{ Sha256, Digest };
use url::Url;

use crate::utils::utils;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct QueryBlockChain {
    pub chain: Vec<Block>,
    pub length: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Transaction {
    pub sender: String,
    pub recipient: String,
    pub amount: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Block {
    pub index: usize,
    pub timestamp: SystemTime,
    pub transactions: Vec<Transaction>,
    pub proof: u64,
    pub prev_hash: String,
}

impl Default for Block {
    fn default() -> Self {
        Block {
            index: Default::default(),
            timestamp: time::UNIX_EPOCH,
            transactions: Default::default(),
            proof: Default::default(),
            prev_hash: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct BlockChain {
    pub chain: Vec<Block>,
    pub current_transactions: Vec<Transaction>,
    pub nodes: HashSet<String>
}


impl Default for BlockChain {
    fn default() -> Self {
        BlockChain {
            chain: vec![],
            current_transactions: vec![],
            nodes: HashSet::new()
        }
    }
}


impl BlockChain {
    pub fn new() -> Self {
        Default::default()
    }
    
    pub fn new_block(&mut self, proof: u64, prev_hash: &str) -> Option<&Block> {
        let block = Block {
            index: self.chain.len() + 1,
            timestamp: SystemTime::now(),
            transactions: self.current_transactions.to_owned(),
            proof,
            prev_hash: prev_hash.to_owned(),
        };
        
        // clear the transaction info
        self.current_transactions.clear();
        
        // append new block to chain
        self.chain.push(block);
        
        // return the last block
        self.chain.last()
    }
    
    pub fn last_block(&self) -> Option<&Block> {
        self.chain.last()
    }
    
    pub fn new_transaction(&mut self, sender: &str, recipient: &str, amount: u64) -> usize {
        let transaction = Transaction {
            sender: sender.to_owned(), recipient: recipient.to_owned(), amount
        };
        self.current_transactions.push(transaction);
        
        // return the length of chains
        self.chain.len() + 1
    }
    
    pub fn hash(block: &Block) -> utils::Result<String>  {
        let block_bytes = serialize(&block)?;
                   
        let mut hasher = Sha256::new();
        hasher.input(block_bytes);
        let hashed_block = format!("{:x}", hasher.result());
        Ok(hashed_block)
    }
    
    // u64::max_value()
    pub fn proof_of_work(&self, last_proof: u64) -> u64 {
        let mut proof = 0u64;
        
        // either use gpu to mine blocks
        if cfg!(feature = "cuda") {
            #[cfg(feature = "cuda")]
            {
                proof = gpu_mining::cuda_sha256(last_proof, "0000").unwrap().into();
            }
        } else {
            while !Self::valid_proof(last_proof, proof) {
                proof += 1;
            }
        }
        
        proof
    }
    
    pub fn register_node<T: AsRef<str>>(&mut self, address: T) -> utils::Result<bool> {
        let parsed_url = Url::parse(address.as_ref())?;
        let host= if parsed_url.has_host() { parsed_url.host_str().unwrap() } else { return Ok(false); };
        
        let netloc = host.to_owned();
        self.nodes.insert(netloc);
        Ok(true)
    }
    
    pub fn valid_chain<T: AsRef<[Block]>>(&self, chain: T) -> bool {
        let mut block_peek = chain.as_ref().iter().peekable();
        while let Some(block) = block_peek.next() {
            let next_block = block_peek.peek();
            if next_block.is_none() { break; }
            // caculate the hash value of each block, and valid it with the current nodes'
            let to_be_verified = Self::hash(&block);
  
            if to_be_verified.map(|v| next_block.unwrap().prev_hash.ne(&v)).is_err() {
                return false;
            }
                    
            if !Self::valid_proof(block.proof, next_block.unwrap().proof) {
                return false;
            }
        }
        
        true
    }
    
    pub fn resolve_conflicts(&mut self) -> utils::Result<bool> {
        let neighbours = &self.nodes;
        let mut new_chain: Vec<Block> = vec![];
        let mut max_length = self.chain.len();
        
        for node in neighbours.iter() {
            let node_url = String::from("http://") + node + ":9000/chain/";          
            let mut response = reqwest::Client::new().get(node_url.as_str()).send()?;
            
            if response.status().eq(&200) {
                let responsed_chain: QueryBlockChain = response.json()?;
                
                let length = responsed_chain.length;
                let chain = &responsed_chain.chain;
                
                if length > max_length && self.valid_chain(chain) {
                    max_length = length;
                    new_chain = chain.clone(); // maybe clone is not a good way
                }
            }
        }
        
        if !new_chain.is_empty() {
            self.chain = new_chain;
            return Ok(true);
        }
        Ok(false)
    }
    
    pub fn valid_proof(last_proof: u64, proof: u64) -> bool {
        let guess = format!("{}{}", last_proof, proof);
        
        let mut hasher = Sha256::new();
        hasher.input(guess.as_bytes());
        let hashed = format!("{:x}", hasher.result());
        hashed.as_str().starts_with("0000")
    }
}