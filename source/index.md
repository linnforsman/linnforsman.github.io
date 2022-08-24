---
layout: default
title: Linn Forsman
---

<div class="site">
<h1>Index</h1>
</div>
<ul class="posts">
  {% for post in site.posts %}
  <li>
    <span>{{ post.date | date_to_string }}</span> /
    <a href="{{ post.url }}">{{ post.title }}</a>
  </li>
  {% endfor %}
</ul>