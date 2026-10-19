# Number Buddy

Number Buddy is a tiny demo project that adds and divides numbers, and can pull
a number from a web API. It exists so the RepoJudge test suite has a small,
predictable repository to analyse.

## Features

- Provides a command line interface for adding two numbers.
- Uses the requests library to fetch a number from a URL.
- Includes a unit test suite that runs with pytest.
- Stores every result in a PostgreSQL database.
- Supports Docker deployment with a one-command build.

## Installation

```
pip install -r requirements.txt
```

## Usage

```
python main.py 2 3
```
