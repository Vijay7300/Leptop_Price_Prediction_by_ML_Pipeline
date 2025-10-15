# this file is created because create directries on heroku

mkdir -p ~/.streamlit/

echo "\
[server]\n\
port = $PORT\N\
enableCORS =false\n\
headless=true\n\
\n\
" > ~/.streamlit/config.toml