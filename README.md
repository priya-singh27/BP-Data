##Docker commands

```
docker build -t bp-image-monitor
```

## Run the docker container locally

```
docker run -p 8000:8000 -e API_KEY=<api_key> -it bp-image-monitor
```

## Rsync

``` 
rsync -av --exclude-from=.gitignore --exclude-from=.dockerignore --exclude='*.jpg' --exclude='*.png' . <username>@<server>:/home/<username>/bpapp/
```
