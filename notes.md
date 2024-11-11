to create new virtual env (note, that if u're using jupiter it will create it's own venv, don't create it by yourself)
```
python3 -m venv <myenvpath>
```

to up the virtual env
```
source .venv/bin/activate
```

to down virtual env
```
diactivate
```

to install dependenses
```
pip install --no-cache-dir -r requirements.txt
```

Login to wandb. Use it BEFORE run your code with wandb
```
wandb login
```
or relogin to another acount
```
wandb login --relogin
```