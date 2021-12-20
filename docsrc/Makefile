# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# command to create a 'docs/' folder containing the documentation for GitHub pages
# docs are created with sphinx and moved to the new folder 'docs/', which is the root folder for GitHub pages
# the command also add the documentation to git
# usage example: make clean html docs
docs:
	@rm -rf ../docs/
	@mkdir -p ../docs/
	@cp -r build/html/. ../docs/
	@git add ../docs/.

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
