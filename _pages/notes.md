---
layout: editorial_page
permalink: /notes/
title: Research notes
eyebrow: Working things out
subtitle: Short explanations of adaptation and efficient inference, with examples you can inspect.
description: Notes on test-time adaptation, matrix tiling, and the Relax-to-TIR compiler pipeline.
nav: true
nav_order: 5
nav_title: Notes
---

<ol class="note-index">
{% assign notes = site.pages | where: 'research_note', true | sort: 'note_order' %}
{% for note in notes %}
  <li><div class="note-index-meta"><span>{{ note.topic }}</span><span>{{ note.reading_time }} min read</span></div><div><h2><a href="{{ note.url | relative_url }}">{{ note.title }}</a></h2><p>{{ note.description }}</p></div></li>
{% endfor %}
</ol>

<p class="editorial-small-note">The <a href="{{ '/lab/' | relative_url }}">interactive lab</a> accompanies the adaptation note.</p>
