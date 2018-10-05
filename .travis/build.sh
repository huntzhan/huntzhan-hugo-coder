echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

# build site (generate ./public)
hugo
# copy ./public to /tmp/hugo-public
mkdir -p /tmp/hugo-public && cp -r ./public/* /tmp/hugo-public
cd /tmp/hugo-public

find .