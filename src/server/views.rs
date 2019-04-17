use actix_web::{ HttpRequest, HttpResponse, AsyncResponder, HttpMessage, Error as HttpResponseErr, FutureResponse };
use futures::{ Future, future::result as FutResult };
use juniper::http::{ graphiql::graphiql_source, GraphQLRequest };
use serde_derive::{ Deserialize, Serialize };
use serde_json::json;
use std::collections::HashSet;

use crate::chains::chain_types::{ self, Transaction };
use crate::utils::utils::NODE_IDENTIFIER;
use super::actor::{ DbState, DbOperation, ReturnType };

const KEY: &str = "chain";


#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Nodes {
    pub nodes: HashSet<String>,
}


pub(crate) fn new_transaction(req: HttpRequest<DbState>) -> FutureResponse<HttpResponse, HttpResponseErr> {
    req.json().from_err()
        .and_then(move |transaction: Transaction| {
            let query = DbOperation::QueryByKey(KEY.to_owned());
            let mut block_chain = req.state().db.send(query).wait();
            
            match block_chain {
                Ok(Ok(ReturnType::BLOCKCHAIN(ref mut ch))) => {
                    let length_of_block_chain = ch.new_transaction(&transaction.sender, &transaction.recipient, transaction.amount);
            
                    let update = DbOperation::UpdateBlockChain(KEY.to_owned(), ch.clone());
                    let _ = req.state().db.send(update).wait();
                    let response = json!({
                        "message": format!("Transaction will be added to Block {}", length_of_block_chain)
                    });
                    Ok(HttpResponse::Ok().json(response))
                }
                _ => Ok(HttpResponse::Ok().json("There's no any bitcoin right now! The transaction will not be finished!"))
            }
    }).responder()
}


pub(crate) fn mine(req: HttpRequest<DbState>) -> FutureResponse<HttpResponse, HttpResponseErr> {
    let query = DbOperation::QueryByKey(KEY.to_owned());
    req.state().db.send(query).from_err().and_then(move |mut chains| {
        match chains {
            Ok(ReturnType::BLOCKCHAIN(ref mut ch)) => {
                let last_block = ch.last_block();
                let last_proof = last_block.map_or(0, |block| block.proof);
                let current_proof = ch.proof_of_work(last_proof);
                // if the blockchain has no block, create an empty hash value
                let prev_hash = last_block.map_or(String::new(), |block| chain_types::BlockChain::hash(block).unwrap());
                
                // send POW and proof
                let chain_index = ch.new_transaction("0", NODE_IDENTIFIER, 1);         
                let latest_block = ch.new_block(current_proof, &prev_hash).unwrap();
                
                let update = DbOperation::UpdateBlock(KEY.to_owned(), latest_block.clone());
                let _ = req.state().db.send(update).wait();
                
                let block = json!(latest_block);
                let response = json!({
                    "message": "New Block Forged",
                    "index": block["index"],
                    "transaction": block["transaction"].as_array(), // 
                    "proof": block["proof"],
                    "prev_hash": block["prev_hash"]
                });
                FutResult(Ok(HttpResponse::Ok().json(response)))
            }
            _ => {
                FutResult(Ok(HttpResponse::Ok().json("There's no coin to be mined!")))
            }
        }
    }).responder()
}


pub(crate) fn chain(req: HttpRequest<DbState>) -> FutureResponse<HttpResponse, HttpResponseErr> {
    let query = DbOperation::QueryByKey(KEY.to_owned());
    req.state().db.send(query).from_err().and_then(move |chains| {
        match chains {
            Ok(ReturnType::BLOCKCHAIN(ref ch)) => {
                let response = json!({
                   "chain":  ch.chain,
                   "length": ch.chain.len()
                });
                FutResult(Ok(HttpResponse::Ok().json(response)))
            }
            _ => {
                let response = json!({
                   "chain":  [],
                   "length": 0
                });
                FutResult(Ok(HttpResponse::Ok().json(response)))
            }
        }
    }).responder()
}


pub(crate) fn register_nodes(req: HttpRequest<DbState>) -> FutureResponse<HttpResponse, HttpResponseErr> {
    req.json().from_err()
        .and_then(move |nodes: Nodes| {
            if nodes.nodes.is_empty() {
                return Ok(HttpResponse::Ok().json("Error: Please supply a valid list of nodes"));
            }
            
            let query = DbOperation::QueryByKey(KEY.to_owned());
            let mut chains = req.state().db.send(query).wait();
            
            if let Ok(Ok(ReturnType::BLOCKCHAIN(ref mut ch))) = chains {
                let _nodes = &nodes.nodes;
                
                // register every node
                // _nodes.iter().map(|node| ch.register_node(node)); // cannot use map due to iterator is a lazy executor
                // _nodes.iter().map(ch.register_node); // the same as above line
                /*
                for node in _nodes.iter() {
                    let r = ch.register_node(node);
                    println!("registering node now: {:?}", r);
                }
                */
                _nodes.iter().for_each(|node|{ let _ = ch.register_node(node); });
                
                // add these nodes to database
                let update = DbOperation::UpdateBlockChain(KEY.to_owned(), ch.clone());
                let _ = req.state().db.send(update).wait();
                
                let response = json!({
                    "message": "New nodes have been added",
                    "total_nodes": ch.nodes.len()
                });
                Ok(HttpResponse::Ok().json(response))
            } else {
                Ok(HttpResponse::Ok().json("Error: Please supply a valid list of nodes"))
            }     
    }).responder()
}


pub(crate) fn consensus(req: HttpRequest<DbState>) -> FutureResponse<HttpResponse, HttpResponseErr> {
    let query = DbOperation::QueryByKey(KEY.to_owned());
    req.state().db.send(query).from_err().and_then(move |mut chains| match chains {
        Ok(ReturnType::BLOCKCHAIN(ref mut ch)) => {
            match ch.resolve_conflicts() {
                Ok(resolved) => { 
                    let response = if resolved {
                        let update = DbOperation::UpdateBlockChain(KEY.to_owned(), ch.clone());
                        let _ = req.state().db.send(update).wait();
                        json!({
                            "message": "Our chain was replaced",
                            "new_chain": ch.chain,
                        })
                    } else {
                        json!({
                            "message": "Our chain is authoritative",
                            "new_chain": ch.chain,
                        })
                    };
                    Ok(HttpResponse::Ok().json(response))
                }
                Err(e) => Ok(HttpResponse::Ok().json(e.description()))
            }
        }
        _ => Ok(HttpResponse::Ok().json("Failed to resolve the block chain!"))
    }).responder()
}