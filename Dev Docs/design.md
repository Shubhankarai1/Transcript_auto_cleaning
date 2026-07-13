# AI Mentor UI Design Guidelines

## Goal

Redesign the application to feel like a modern SaaS product while keeping the implementation simple and fully compatible with Streamlit.

Design Principles:
- Minimal
- Clean
- Centered layouts
- Plenty of whitespace
- Consistent spacing
- No unnecessary animations or complex components

---

# Theme

Background: #FAFAFA

Cards: White

Primary Color: #2563EB

Accent Color: #14B8A6

Border Radius: 12px

Use soft shadows for cards.

Maximum content width:
1000px

Center all pages.

---

# 1. Landing Page

Purpose:
Introduce the AI Mentor before login.

Sections:

## Hero

Title:
AI Mentor

Subtitle:
Your personalized AI learning companion powered by IITM course material.

Buttons:
- Get Started
- Login

---

## Features

Display 3 simple cards.

- Personalized Learning
- AI Mentor
- Progress Tracking

Each card contains:
- Icon
- Title
- One-line description

---

## How It Works

Simple 4-step horizontal section.

Create Account
↓

Complete Profile
↓

Get Learning Roadmap
↓

Start Learning

---

Footer

Simple footer with:
- IITM Capstone Project
- Version

---

# 2. Login Page

Use a centered card.

Include:
- Logo / Title
- Email
- Password
- Login Button
- Sign Up link

Keep maximum card width around 450px.

Avoid full-width forms.

---

# 3. Onboarding

Replace wide forms with centered cards.

Maximum width:
700px

Each step inside one card.

Show progress indicator.

Example:

Step 2 of 3

██████░░░░

---

Only ask essential information.

Step 1
- Full Name
- Industry

Step 2
- Years of Experience
- Career Goal

Step 3
- AI Learning Goals
- Weekly Learning Hours

Buttons:
Back | Next

Use dropdowns or radio buttons wherever possible.

Avoid large empty text areas.

---

# 4. AI Mentor Page

Layout:

Left Sidebar

- User Name
- Edit Profile
- Sign Out

Navigation:
- AI Mentor

---

Main Area

Large page title:

Welcome, {User Name}

Small subtitle.

Below it:

Chat Interface

Use Streamlit chat components.

Remove long introductory paragraphs.

Instead show one welcome message.

Example:

Welcome!
Ask me anything about your course.

Below this:
Chat history.

Bottom:
Chat input.

---

# Typography

Use clear hierarchy.

Page Title

Large

Section Heading

Medium

Body

Normal

Keep text concise.

---

# Spacing

Use generous spacing between sections.

Avoid stretching components across the entire screen.

Keep all important content centered.

---

# General Rules

- Use cards for grouping content.
- Avoid full-width forms.
- Maintain consistent button styles.
- Keep pages clean and uncluttered.
- Prioritize readability over decorative UI.
- Do not introduce unnecessary animations or complex visual effects.
- Keep implementation simple and Streamlit-friendly.