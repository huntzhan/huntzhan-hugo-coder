echo -e "\033[0;32mDeploying updates to GitHub...\033[0m"

cd /tmp/hugo-public

git config --global user.email "travis@travis-ci.org"
git config --global user.name "Travis CI"

git init
git add --all
git commit -m "Published By Travis CI."

git remote add origin https://${GH_PAT}@github.com/huntzhan/huntzhan.github.io.git
git push -u origin master --force