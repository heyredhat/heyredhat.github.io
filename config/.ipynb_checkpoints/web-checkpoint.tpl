{% extends 'full.tpl'%}
<!DOCTYPE html>
<html lang="en">
<head>
    {% block head %}
    <title>{% block title %}{% endblock %}</title>
    {% endblock %}
</head>
<body>
    <div id="content">
        {% block content %}
        {% endblock %}
    </div>
    <div id="footer">
        {% block footer %}
        <div id="logo">
            <a href="index.html"><img src="img/logo.png"/></a>
        </div>
        {% endblock %}
    </div>
</body>
</html>