"""docs/PBT.md against the code: the config reference lists every PbtCfg field with its default,
every built-in mutation function is documented, the examples construct and validate, and the
links resolve."""

import os
import re
from dataclasses import MISSING, fields

import pytest
import yaml

from rl_games.common.pbt import PbtCfg, validate
from rl_games.common.pbt.mutation import MUTATION_FUNCS

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs")
PBT_MD = os.path.join(DOCS, "PBT.md")
FIELDS = [f.name for f in fields(PbtCfg)]


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def section(text, heading):
    """The body of the `## <heading>` section of a markdown text, up to the next `## ` heading."""
    match = re.search(rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text, re.M | re.S)
    assert match, f"no '## {heading}' section"
    return match.group(1)


def reference_rows():
    """[(key, default cell)] of the config reference table, in order: | `key` | `default` | ..."""
    return re.findall(r"^\| `(\w+)` \| `([^`]*)` \|", section(read(PBT_MD), "Config reference"), re.M)


def field_default(name):
    f = next(f for f in fields(PbtCfg) if f.name == name)
    default = f.default if f.default is not MISSING else f.default_factory()
    return list(default) if isinstance(default, tuple) else default  # YAML has lists, not tuples


def code_blocks(language):
    return re.findall(rf"^```{language}\n(.*?)^```", read(PBT_MD), re.M | re.S)


def pbt_sections():
    """The `pbt` sections of the YAML examples."""
    loaded = [yaml.safe_load(block) for block in code_blocks("yaml")]
    return [doc["pbt"] for doc in loaded if isinstance(doc, dict) and "pbt" in doc]


# ---------------------------------------------------------------- config reference

def test_config_reference_lists_the_fields_in_order():
    assert [key for key, _ in reference_rows()] == FIELDS


@pytest.mark.parametrize("name", FIELDS)
def test_config_reference_shows_the_default(name):
    documented = yaml.safe_load(dict(reference_rows())[name])
    default = field_default(name)
    assert documented == default and type(documented) is type(default), (
        f"docs/PBT.md shows {name} default {documented!r}, PbtCfg has {default!r}")


# ---------------------------------------------------------------- mutation functions

def test_every_built_in_mutation_function_is_documented():
    # tests may register custom functions: the built-ins are the ones mutation.py defines
    built_in = sorted(name for name, fn in MUTATION_FUNCS.items() if fn.__module__ == "rl_games.common.pbt.mutation")
    table = section(read(PBT_MD), "Mutation functions")
    documented = re.findall(r"^\| `(\w+)` \|", table, re.M)
    assert sorted(documented) == built_in


# ---------------------------------------------------------------- examples

def test_yaml_examples_construct():
    sections = pbt_sections()
    assert len(sections) >= 2
    for pbt in sections:
        PbtCfg.from_dict(pbt)


def test_usage_example_validates_against_a_ppo_config(capsys):
    params = yaml.safe_load(read(os.path.join(REPO, "rl_games", "configs", "ppo_cartpole.yaml")))["params"]
    params["config"]["lr_schedule"] = "adaptive"   # the example mutates kl_threshold
    examples = [pbt for pbt in pbt_sections() if pbt.get("mutation")]
    assert examples
    for pbt in examples:
        cfg, mutable = validate({"params": params, "pbt": pbt})
        assert set(mutable) == set(pbt["mutation"]) and cfg.objective
    assert "WARNING" not in capsys.readouterr().out


def test_python_examples_compile_and_import():
    blocks = code_blocks("python")
    assert blocks
    for block in blocks:
        compile(block, "docs/PBT.md", "exec")
        for line in block.splitlines():
            if line.startswith(("from rl_games", "import rl_games")):
                exec(line, {})


# ---------------------------------------------------------------- links

def github_anchor(heading):
    """The anchor GitHub gives a heading: lowercase, punctuation dropped, spaces to hyphens."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def check_links(text, base_dir):
    prose = re.sub(r"^```.*?^```", "", text, flags=re.M | re.S)
    anchors = {github_anchor(h) for h in re.findall(r"^#{1,6} (.+)$", prose, re.M)}
    targets = re.findall(r"\]\(([^)\s]+)\)", prose)
    assert targets
    for target in targets:
        if target.startswith(("http://", "https://")):
            continue
        if target.startswith("#"):
            assert target[1:] in anchors, f"no heading for the link {target}"
        else:
            path = os.path.normpath(os.path.join(base_dir, target.split("#")[0]))
            assert os.path.exists(path), f"the link {target} names no file"


def test_pbt_md_links_resolve():
    check_links(read(PBT_MD), DOCS)


def test_readme_pbt_section_links_resolve():
    check_links(section(read(os.path.join(REPO, "README.md")), "Population Based Training"), REPO)
