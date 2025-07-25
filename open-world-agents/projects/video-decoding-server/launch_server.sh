export DATA_DIR=${1}

# if not given, raise error
if [ -z "$DATA_DIR" ]; then
    echo "Usage: $0 <data_dir>"
    exit 1
fi

docker compose up -d --build --force-recreate