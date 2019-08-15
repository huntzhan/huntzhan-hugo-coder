# Append timestamps.
python .travis/md_append_timestamp.py content/posts/ml/math.md content/posts/ml/math.md

# build site (generate ./public)
hugo
# copy ./public to /tmp/hugo-public
mkdir -p /tmp/hugo-public && cp -r ./public/* /tmp/hugo-public
cd /tmp/hugo-public

find .
if [ ! -f index.html ]; then
    echo "index.html not found!"
    exit 1
fi
exit 0