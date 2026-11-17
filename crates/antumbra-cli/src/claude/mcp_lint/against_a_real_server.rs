use anyhow::Context;

use super::*;

#[test]
#[ignore = "needs ANTUMBRA_TOOLS_LIST, a captured tools/list result"]
fn report_what_a_live_server_returns() -> anyhow::Result<()> {
    let Ok(path) = std::env::var("ANTUMBRA_TOOLS_LIST") else {
        println!("ANTUMBRA_TOOLS_LIST unset -- skipped");
        return Ok(());
    };
    let raw = std::fs::read_to_string(&path).context("read the capture")?;
    let answer: Value = serde_json::from_str(&raw).context("parse the capture")?;
    let tools = tools_in(&answer).context("a tools/list answer")?;
    let findings = lint(&tools);
    println!("\n{}", render("antumbra", tools.len(), &findings));
    println!(
        "\n  {} tool(s), {} finding(s), {} failure(s), {} shape",
        tools.len(),
        findings.len(),
        failures(&findings),
        shaped(&findings)
    );
    assert_eq!(
        failures(&findings),
        0,
        "antumbra's own surface must not carry a schema the API refuses"
    );
    Ok(())
}
