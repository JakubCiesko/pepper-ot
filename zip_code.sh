#!/bin/bash

echo "Preparing tmp folder"
rm -rf /tmp/thesis_submit/code
mkdir -p /tmp/thesis_submit/code

rsync -avRr --files-from=code_include.list ./ /tmp/thesis_submit/code/

cd /tmp/thesis_submit
zip -r code.zip code
unzip -t code.zip
echo "Done, look in /tmp/thesis_submit/code"
