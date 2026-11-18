import json

from build_overview import OUT_JSON
from northwind import ROOT
from q1_growth_drivers import OUT_JSON as Q1_JSON
from q2_product_risk import OUT_JSON as Q2_JSON
from q3_relationship_selling import OUT_JSON as Q3_JSON

TEMPLATE = ROOT / "site" / "template.html"
ARTIFACT_OUT = ROOT / "site" / "artifact.html"
SITE_OUT = ROOT / "docs" / "index.html"


def render(template_path=TEMPLATE, data_path=OUT_JSON):
    data = json.loads(data_path.read_text())
    data["q1"] = json.loads(Q1_JSON.read_text())
    data["q2"] = json.loads(Q2_JSON.read_text())
    data["q3"] = json.loads(Q3_JSON.read_text())
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return template_path.read_text().replace("/*__DATA__*/", payload)


def wrap_standalone(fragment):
    split = fragment.index("</style>") + len("</style>")
    head, body = fragment[:split], fragment[split:]
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, "
        "initial-scale=1, viewport-fit=cover\">\n"
        "<meta name=\"description\" content=\"Three business questions "
        "answered from the Northwind database: growth drivers, product "
        "risk and dedicated sellers.\">\n"
        + head + "\n</head>\n<body>\n" + body.strip()
        + "\n</body>\n</html>\n"
    )


def main():
    fragment = render()
    ARTIFACT_OUT.write_text(fragment)
    SITE_OUT.parent.mkdir(parents=True, exist_ok=True)
    SITE_OUT.write_text(wrap_standalone(fragment))
    (SITE_OUT.parent / ".nojekyll").touch()
    print(f"Wrote {SITE_OUT} (and {ARTIFACT_OUT} for the Claude preview)")


if __name__ == "__main__":
    main()
