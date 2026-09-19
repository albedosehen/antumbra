//! Test seams (feature `testing`): a transport answering from canned routes,
//! and a throwaway RSA key pair standing in for an App's. The key is
//! test-only and never a real credential.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value;

use crate::api::{ApiError, GithubTransport, HttpResponse};

/// A throwaway RSA-2048 private key (PKCS#8 PEM) for signing App tokens in
/// tests.
pub const APP_PRIVATE_KEY_PEM: &[u8] = b"-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQDMRax89LGXOtI6
2/wNDH9gV2AvSTetDqsPIAX0JsCVRWt/XqWm/F/mJrfbgsmjlLVWy8PJWl0+hdtu
wo93naG4DyWJW+8AEX8AL1HjrPBimWe3+qXBFxvPVygNqT2PvGYmO7X+jtHXEKvR
aVcOnOByukLaXuSQVaRa40VFjBIU0WKBlU72apuPAtoB9taZ6roz7rJ8iTGE4DYt
0LflF5LjQb6IjANfEAPcZICt0ufzD6OgG+1peuhqgwZErWfQgZ9eOOT2O9SdEB5z
nc/l1LR/F94WhOO29JTGohVN46gdcEkS9/FGBBi5/4p3L74LcpJPAnIKWv2jpD2n
q0NVpW13AgMBAAECggEAUHJ4FdYAQsDFnqyYPUNYvsZqePTq2lrWf2RrM9Y3LhJi
3YyWzIbD9c31xpthcezU5dPlzVyrMD5jRuGUwtTvpZ9BdzEflPVPAPGh3Hp1ST+F
G2247ax+JU/71DV8qyjVSeVmLVRty7cjE5vaz0R1GHnGbl3EwhsYWTr8QwGA9XU0
qWwDnBcBDpSuqM3S3FgbfE6xKzOtEIsmtwwVHi112qlWcSXFrSSFwm7l5/0huURV
6dAyeugDUWvZTjTZ/+liCsmpEbw4ftxeMLMONaDvQIQN2fgBvvGCBwJI/8PnV3vG
nGI//3ECD9ru1pjsqLngcA8ugpoT+FpN4IG5WLJ94QKBgQDkvIUbmLn9TkrvVgki
FHoOl3Y1uVYqYnYhaKc1GigMU2VJkW5NP6m19KEw13vkdc+KwZEzzxZIN1LmwCPc
TfFlqjCFs7/Urtvvnl3bnw1o2KrhXiR1JQrcoLDzwkid8Q6mEQ66MiEgDkBjLaOu
rkWOGByBgwujTDEkxsbv9XlJvwKBgQDknqvkppDjEoozBfH+S13n7qebJ3z6hAgN
fOgqUN8FTRO7gHebE0wCz9AVbXMvduv0oAiW77pmYUIl4otoWc3M/klcRTB1BOix
Eu5vD/oCu9ZHoCMO57yFOIa+YNjryer9NLm9ow/jTC5BUYYn5/jQJGnkNjWZM57q
Wh8Vy+4aSQKBgQDQNCFdG0nAnnFrJX8uvEDV41xATrF15yXsBxycI3Dst0RtEKm8
OwS5kTDgCmTFcc82WDdZV1jK50DYtXBu6aufhKiiKxmj+H5NwHNio4ZLN11jwpOg
5dTbOpGXb/M1gOR6mPA038hzK0XEgRiKuiqpypy37pa7T3E0LpOKfICodQKBgQDI
OkemDFPc7EHpig11gCCQny5f7ufAqJ484eacGRQamnTrxQn74Zyy4bsG6UL2kRr6
tqaPOwpv3EKI167tB6n9HcC2dUqJUnFRlJkK4F1Aw65aMOBDj6ZGr0kjt8KET+Xl
OaZrdkLV+cSRJItwq/P4p8uuOeQbd2B5M9EB0AeLMQKBgGQVVr40SZlQmMntXszm
0otGkCBvq5kMmhPFWt+zM8ZeAjf9lOIiYdrDNxje2HpK+6nzfLLwTwdwh31bWcvV
NYpgHMsZXpcBVVAT3Nm6eP8OaPsCB/tzMfRWN5GlrqoqprsIl88tSnpmvQ0KVS9l
nDEh0mKr23w08IOzqgciZR8L
-----END PRIVATE KEY-----
";

/// The matching public key.
pub const APP_PUBLIC_KEY_PEM: &[u8] = b"-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEAzEWsfPSxlzrSOtv8DQx/
YFdgL0k3rQ6rDyAF9CbAlUVrf16lpvxf5ia324LJo5S1VsvDyVpdPoXbbsKPd52h
uA8liVvvABF/AC9R46zwYplnt/qlwRcbz1coDak9j7xmJju1/o7R1xCr0WlXDpzg
crpC2l7kkFWkWuNFRYwSFNFigZVO9mqbjwLaAfbWmeq6M+6yfIkxhOA2LdC35ReS
40G+iIwDXxAD3GSArdLn8w+joBvtaXroaoMGRK1n0IGfXjjk9jvUnRAec53P5dS0
fxfeFoTjtvSUxqIVTeOoHXBJEvfxRgQYuf+Kdy++C3KSTwJyClr9o6Q9p6tDVaVt
dwIDAQAB
-----END PUBLIC KEY-----
";

/// A transport answering from canned routes (`"GET <url>"` and
/// `"POST <url>"`) and recording every call. An unmatched route answers 404
/// with a body naming it, so a test sees what it forgot.
#[derive(Default)]
pub struct FakeTransport {
    routes: HashMap<String, (u16, Vec<u8>)>,
    calls: Mutex<Vec<String>>,
    bearers: Mutex<Vec<String>>,
}

impl FakeTransport {
    pub fn new() -> Self {
        Self::default()
    }

    /// Answer `method url` with `status` and `body`.
    pub fn route(mut self, method: &str, url: &str, status: u16, body: impl Into<Vec<u8>>) -> Self {
        self.routes
            .insert(format!("{method} {url}"), (status, body.into()));
        self
    }

    /// Answer `method url` with `status` and a JSON body.
    pub fn json(self, method: &str, url: &str, status: u16, body: &Value) -> Self {
        let bytes = serde_json::to_vec(body).expect("serializable");
        self.route(method, url, status, bytes)
    }

    /// Every call so far, as `"METHOD url"`, in order.
    pub fn calls(&self) -> Vec<String> {
        self.calls.lock().expect("calls").clone()
    }

    /// The bearer presented on each call, in order.
    pub fn bearers(&self) -> Vec<String> {
        self.bearers.lock().expect("bearers").clone()
    }

    fn answer(&self, key: String, bearer: &str) -> Result<HttpResponse, ApiError> {
        self.calls.lock().expect("calls").push(key.clone());
        self.bearers
            .lock()
            .expect("bearers")
            .push(bearer.to_string());
        Ok(match self.routes.get(&key) {
            Some((status, body)) => HttpResponse {
                status: *status,
                body: body.clone(),
            },
            None => HttpResponse {
                status: 404,
                body: format!("no canned route for {key}").into_bytes(),
            },
        })
    }
}

impl GithubTransport for FakeTransport {
    fn get(&self, url: &str, bearer: &str, _accept: &str) -> Result<HttpResponse, ApiError> {
        self.answer(format!("GET {url}"), bearer)
    }

    fn post_json(&self, url: &str, bearer: &str, _body: &Value) -> Result<HttpResponse, ApiError> {
        self.answer(format!("POST {url}"), bearer)
    }
}
