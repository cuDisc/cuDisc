# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# -- Project information -------------------------------------------------------

project = 'cuDisc'
copyright = '2024, Alfie Robinson, Richard Booth'
author = 'Alfie Robinson, Richard Booth'
release = '1.0'

# -- General configuration -----------------------------------------------------

extensions = [
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
]

autosectionlabel_prefix_document = True

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output ---------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = []

html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
}

html_context = {
    'display_github': True,
    'github_user': 'cuDisc',
    'github_repo': 'cuDisc',
    'github_version': 'main',
    'conf_py_path': '/docs/',
}

# -- MathJax -------------------------------------------------------------------

mathjax3_config = {
    'tex': {
        'macros': {
            'St': r'\mathrm{St}',
            'vfrag': r'v_\mathrm{frag}',
        }
    }
}

# -- Intersphinx ---------------------------------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}
