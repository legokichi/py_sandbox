# memo
* https://pipenv-ja.readthedocs.io/ja/translate-ja/
* http://pipenv-ja.readthedocs.io/ja/translate-ja/basics.html
* do not use anaconda.
* use system python.

## install

install pyenv

```sh
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
cat <<'EOF'>>~/.bash_profile
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
if command -v pyenv 1>/dev/null 2>&1; then
  eval "$(pyenv init -)"
fi
EOF
pyenv install --list
```

install pipenv (with system python)

```sh
pip install pipenv
```

## setup

```sh
mkdir -p pipenv_sandbox
cd pipenv_sandbox
cat <<'EOF'>>Pipfile
[[source]]
url = "https://pypi.python.org/simple"
verify_ssl = true
name = "pypi"

[packages]

[dev-packages]

EOF
pipenv install
pipenv update --outdated
pipenv update
pipenv --python 3.6.5
pipenv shell
```


## repl

pyenv + anaconda way

```sh
pyenv install anaconda3-5.2.0
pyenv shell anaconda3-5.2.0
python
```

pipenv way

```sh
pipenv shell
pipenv run python
```

