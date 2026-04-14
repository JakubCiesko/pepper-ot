BASE_DIR=$(PEPPER_ROOT)
MYAPPS_DIR=$(PEPPER_ROOT)/apps_dir
INSTALL_PKG=$(MYAPPS_DIR)/install_pkg.py
QIPKG=PATH="$(BASE_DIR)/pyenv/bin:$(PATH)" qipkg
REMOTE_USER=$(FI_MUNI_USER)
REMOTE_HOST=$(FI_THESIS_HOST)
REMOTE_DIR=/home/$(FI_MUNI_USER)/pepper
REMOTE_INSTALL_CMD=$(FI_THESIS_REMOTE_INSTALL_CMD)

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

deploy: pkg
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
	pkgfilename="`basename \"$$pkgfile\"`"; \
	echo "Uploading $$pkgfilename to $(REMOTE_HOST)..."; \
	scp "$$pkgfile" $(REMOTE_USER)@$(REMOTE_HOST):$(REMOTE_DIR)/; \
	echo "Installing on remote machine..."; \
	ssh $(REMOTE_USER)@$(REMOTE_HOST) "$(REMOTE_INSTALL_CMD) $(REMOTE_DIR)/$$pkgfilename"; \
	echo "Deployment complete"
