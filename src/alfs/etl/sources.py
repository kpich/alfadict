"""Source registry for MediaWiki corpus dumps."""

from dataclasses import dataclass


@dataclass
class Source:
    name: str
    dump_url: str
    dump_filename: str
    base_url: str


SOURCES: dict[str, Source] = {
    "wikibooks": Source(
        name="wikibooks",
        dump_url="https://dumps.wikimedia.org/enwikibooks/latest/enwikibooks-latest-pages-articles.xml.bz2",
        dump_filename="enwikibooks-latest-pages-articles.xml.bz2",
        base_url="https://en.wikibooks.org/wiki/",
    ),
    "wikisource": Source(
        name="wikisource",
        dump_url="https://dumps.wikimedia.org/enwikisource/latest/enwikisource-latest-pages-articles.xml.bz2",
        dump_filename="enwikisource-latest-pages-articles.xml.bz2",
        base_url="https://en.wikisource.org/wiki/",
    ),
}
