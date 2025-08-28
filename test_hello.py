from hello import say_hello, add

def test_say_hello():
    assert (
        say_hello("Claire")
        == "Hello, Claire, welcome to Data Engineering Systems (IDS 706)!"
    )

def test_add():
    assert add(2, 3) == 5