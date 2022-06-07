# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import glob
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../ssc_scoring'))


# -- Project information -----------------------------------------------------

project = 'ssc_scoring'
copyright = '2021, Jingnan Jia'
author = 'Jingnan Jia'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.todo',
              'sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              "sphinx.ext.intersphinx",
              "sphinx.ext.mathjax",
              "sphinx.ext.autosectionlabel",
              "matplotlib.sphinxext.plot_directive"
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = []
scripts_files = glob.glob("../../ssc_scoring/*.py")
scripts_files = [os.path.abspath(f) for f in scripts_files]
exclude_patterns = ['dataset',
                    'image_samples',
                    'results',
                    'persistent_cache',
                    '*.mha'] + scripts_files
print(f'excluded patterns: {exclude_patterns}')

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'  #  sphinx_rtd_them will not be available since Sphinx-6.0
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    "external_links": [{"url": "https://github.com/Jingnan-Jia/ssc_scoring", "name": "GitHub"}],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Jingnan-Jia/ssc_scoring",
            "icon": "fab fa-github-square",
        },
    ],
    "collapse_navigation": True,
    "navigation_depth": 3,
    "show_toc_level": 1,
    "footer_items": ["copyright"],
    "navbar_align": "content",
}
html_context = {
    "github_user": "Jingnan Jia",
    "github_repo": "ssc_scoring",
    "github_version": "master",
    "doc_path": "docs/",
    "conf_py_path": "/docs/",
}
html_scaled_image_link = False
html_show_sourcelink = True
# html_favicon = "../images/favicon.ico"
# html_logo = "../images/MONAI-logo-color.png"
html_sidebars = {"**": ["search-field", "sidebar-nav-bs"]}
pygments_style = "sphinx"


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
autodoc_inherit_docstrings = False