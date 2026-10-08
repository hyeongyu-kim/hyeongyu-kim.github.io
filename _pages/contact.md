---
layout: editorial_page
permalink: /contact/
title: Get in touch
eyebrow: Research & professional enquiries
subtitle: Email, profiles, and a contact card you can keep or share.
description: Contact Hyeongyu Kim and download a QR business card or vCard.
contact_script: true
---

<div class="contact-email"><a href="mailto:{{ site.data.socials.email }}">{{ site.data.socials.email }}</a><button type="button" id="copy-email" class="quiet-button" data-email="{{ site.data.socials.email }}">Copy email</button></div>
{% include profile_links.liquid %}

<section class="contact-card-section" aria-labelledby="contact-card-title">
  <h2 id="contact-card-title">A card for your next conversation</h2>
  <p>Scan the QR code to open this site. The contact file adds my name, email, company, and profile links to your address book.</p>
  <img class="contact-card-image" src="{{ '/assets/contact/Hyeongyu_Kim_card.png' | relative_url }}" width="1800" height="1000" alt="Contact card for Hyeongyu Kim: Compiler Engineer at Hyundai Motor Company, test-time adaptation and medical imaging, email khg4309@naver.com, homepage hyeongyu-kim.github.io, and a QR code linking to the homepage.">
  <div class="contact-downloads"><a href="{{ '/assets/contact/Hyeongyu_Kim_card.png' | relative_url }}" download="Hyeongyu_Kim_card.png"><i class="fa-regular fa-image" aria-hidden="true"></i> Download card (PNG)</a><a href="{{ '/assets/contact/Hyeongyu_Kim.vcf' | relative_url }}" download="Hyeongyu_Kim.vcf"><i class="fa-regular fa-address-card" aria-hidden="true"></i> Save contact (vCard)</a><a href="{{ '/assets/contact/homepage-qr.png' | relative_url }}" download="Hyeongyu_Kim_QR.png">QR code only</a><button type="button" id="share-profile" class="quiet-button" data-url="{{ '/' | absolute_url }}">Share profile</button></div>
  <p id="contact-status" class="lab-caption" role="status" aria-live="polite"></p>
</section>
