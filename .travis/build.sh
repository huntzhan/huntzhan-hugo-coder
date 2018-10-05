echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# build site (generate ./publish)
hugo
# copy ./publish to /tmp/publish
mkdir -p /tmp/publish && cp -r ./publish /tmp/publish
cd /tmp/publish

ls -al