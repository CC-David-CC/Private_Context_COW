"""RAM A/B/A control using the same exact-oracle, durable-publication runner.

Every activation restores a retained host snapshot into private GPU pages.
The parent GPU image remains held solely as an unchanged-byte audit witness;
this control does not claim minimal total GPU residency or GPU page sharing.
"""
from cow_speed_8k import main


if __name__ == '__main__':
    main((('host-ram', 'A'), ('host-ram', 'B'), ('host-ram', 'A')))
