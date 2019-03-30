use actix_web::{ server, App, middleware, 
                 http::{ self, NormalizePath },
                 middleware::session::{ SessionStorage, CookieSessionBackend }
};
use actix_redis::RedisSessionBackend;
use std::env;

use crate::{ chains::chain_types, server::{ views, actor }, utils::utils };

pub(crate) fn run() -> utils::Result<()> {
    let config = utils::server_config()?;
    
    let address = config["address"].as_str().unwrap();
    let port = config["port"].as_integer().unwrap();
    let mut workers = config["workers"].as_integer().unwrap() as usize;
    let log_level = config["log"].as_str().unwrap();

    env::set_var("RUST_LOG", format!("actix_web={}", log_level)); // log level
    env_logger::init(); // init a log
    
    let sys = actix::System::new("yt-chain"); // start a system
    workers = num_cpus::get();
    
    let db = match utils::db_config() {
        Ok(db) => db,
        _ => panic!("cannot connect to the database"),
    };
    
    let addr = actix::SyncArbiter::start(workers, move || actor::DBPool{ conn: db.clone() });
    
    let chain_server = server::new(move || {
        vec![
            App::with_state(actor::DbState{ db: addr.clone() })
                .middleware(middleware::Logger::default())
                .middleware(SessionStorage::new(
                    // redis is running on local server with port 6379
                    // RedisSessionBackend::new("127.0.0.1:6379", &[0;32])
                    CookieSessionBackend::signed(&[0; 32])
                    // .cookie_secure(false)
                    .secure(false)
                    // .cookie_max_age(chrono::Duration::days(30)) // session will expire after 30 days
                    .max_age(chrono::Duration::days(30)) // session will expire after 30 days
                ))
                .scope("/", |scope| {
                    scope.default_resource(|r| r.h(NormalizePath::default())) // normalize the path
                    .resource("/new_transaction/new/", |r| {
                        r.method(http::Method::POST).with(views::new_transaction);
                    })
                    .resource("/mine/", |r| {
                        r.method(http::Method::POST).with(views::mine);
                    })
                    .resource("/gpu_mine/", |r| {
                        r.method(http::Method::POST).with(views::gpu_mine);
                    })
                    .resource("/chain/", |r| {
                        r.method(http::Method::GET).with(views::chain);
                    })
                    .resource("/nodes/register/", |r| {
                        r.method(http::Method::POST).with(views::register_nodes);
                    })
                    .resource("/nodes/resolve/", |r| {
                        r.method(http::Method::GET).with(views::consensus);
                    })
                })
        ]
    });
    
    // chain_server.bind_ssl(format!("{}:{}", &address, &port), utils::load_ssl())?.start();
    chain_server.bind(format!("{}:{}", &address, &port))?.start();
    println!("Started http server: {}", format!("{}:{}", &address, &port));
    let _ = sys.run();
    Ok(())
}