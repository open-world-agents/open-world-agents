gpu_args="--gpus all --ipc=host"
alias docker-gpu="docker run $gpu_args -it --rm -v .:/workspace --workdir /workspace"
alias docker-gpu-network="docker run $gpu_args --network=host -it --rm -v .:/workspace --workdir /workspace"
alias docker-ubuntu="docker run -it --rm -v .:/workspace -w /workspace ubuntu:latest"

alias wget_="aria2c -s 16 -x 16"
alias cp_='rsync -a --partial --info=progress2'
alias mv_='rsync -a --partial --info=progress2 -remove-source-files'
rm_() {
    for target in "$@"; do
        [ -e "$target" ] && find "$target" -delete -print | tqdm --desc "Deleting $target" --unit files > /dev/null
    done
}
tar_() {
    local dir="$1"
    # default outfile is $1.tar.gz
    local outfile="$2"
    if [ -z "$outfile" ]; then
        outfile="$dir.tar.gz"
    fi
    local BYTES="$(du -sb "$dir" | cut -f1)"

    tar -cf - "$dir" \
        | tqdm --bytes --total "$BYTES" --desc Processing | gzip \
        | tqdm --bytes --total "$BYTES" --desc Compressed --position 1 \
        > "$outfile"
}
untar_() {
    local infile="$1"
    # default outdir is current directory
    local outdir="${2:-.}"
    local BYTES
    BYTES="$(du -sb "$infile" | cut -f1)"
    mkdir -p "$outdir"
    if file -b --mime-type "$infile" | grep -q 'gzip'; then
        cat "$infile" | tqdm --bytes --total "$BYTES" --desc Decompressing | gunzip | tar -xf - -C "$outdir"
    else
        cat "$infile" | tqdm --bytes --total "$BYTES" --desc Extracting | tar -xf - -C "$outdir"
    fi
}