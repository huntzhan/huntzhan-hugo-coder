echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

hugo
mkdir -p /tmp/publish
cp -r /publish /tmp/publish
cd /tmp/publish
tree