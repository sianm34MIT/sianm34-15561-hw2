"""Run every analysis step and rebuild the site, in order."""
import build_overview
import build_site
import q1_growth_drivers
import q2_product_risk
import q3_relationship_selling


def main():
    """Rebuild all JSON outputs and the site.

    Args:
        None.

    Returns:
        None. Writes ``site/data/*.json`` and ``docs/index.html``.
    """
    build_overview.main()
    q1_growth_drivers.main()
    q2_product_risk.main()
    q3_relationship_selling.main()
    build_site.main()


if __name__ == "__main__":
    main()
