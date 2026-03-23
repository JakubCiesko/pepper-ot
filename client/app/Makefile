MYAPPS_DIR=$(HOME)/Work/school/pepper/apps_dir
BASE_DIR=$(PEPPER_ROOT)
INSTALL_PKG=$(BASE_DIR)/apps_dir/install_pkg.py
QIPKG=PATH="$(BASE_DIR)/pyenv/bin:$(PATH)" qipkg

pkg:
	umask 002; \
	pkgname="$(wildcard *.pml)"; \
	if [ -z "$$pkgname" ]; then \
		echo "I need a .pml file here"; \
		exit 1; \
	fi; \
	basename="`basename -s .pml \"$$pkgname\"`"; \
	rm -fv "$$basename"*.pkg || true; \
	$(QIPKG) bump-version manifest.xml; \
	$(QIPKG) make-package "$$pkgname"; \
	version=`expr $$basename*.pkg : $$basename-'\([.0-9]*\)'`; \
	mv -v "$$basename"*.pkg $(MYAPPS_DIR)/"$$basename"-$$version.pkg || true

install:
	umask 002; \
	pkgname="$(wildcard *.pml)"; \
	if [ -z "$$pkgname" ]; then \
		echo "I need a .pml file here"; \
		exit 1; \
	fi; \
	basename="`basename -s .pml \"$$pkgname\"`"; \
	pkgfile="`ls -t $(MYAPPS_DIR)/\"$$basename\"*pkg|head -n1`"; \
	if [ -z "$$pkgfile" ]; then \
		echo "Package file for $$basename not found in $(MYAPPS_DIR)"; \
		exit 1; \
	fi; \
	$(INSTALL_PKG) "$$pkgfile"
