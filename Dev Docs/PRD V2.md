# **PRD v2: AI Learning Mentor for Mid-Career Professionals**

## **Project Overview**

AI Learning Mentor is a personalized learning and career development platform designed for mid-career and senior professionals who want to become AI-ready and remain competitive in an AI-driven workplace.

Unlike traditional learning platforms that provide static learning paths, the AI Learning Mentor personalizes the learning experience through user profiling, AI readiness assessment, adaptive learning roadmaps, and an AI Mentor that provides contextual educational guidance.

A major component of the platform—the AI Knowledge Layer—has already been developed through a previous IITM Transcript-Based RAG System. This project focuses on transforming that existing AI capability into a complete end-user learning platform.

---

# **Problem Statement**

## **External Problem**

Mid-career and senior professionals lack a structured and personalized approach to learning AI that aligns with their background, career goals, and existing skill levels.

## **Internal Problem**

Professionals often struggle to determine:

* Where to begin learning AI  
* What topics are most relevant to their role  
* Whether they are making meaningful progress  
* How to apply AI effectively within their profession

---

# **Solution**

Develop an AI Learning Mentor that combines:

* Personalized learner profiling  
* AI readiness assessment  
* Personalized learning roadmaps  
* AI-powered educational mentoring

into a unified learning experience.

Rather than functioning solely as a chatbot, the platform acts as an intelligent learning companion that guides professionals throughout their AI learning journey.

---

# **Target Users**

Primary Users

* Mid-career professionals (5–15 years experience)  
* Senior managers and leaders  
* Professionals transitioning into AI-enabled roles  
* Individuals seeking structured AI upskilling

---

# **Existing Foundation (Already Built)**

A significant portion of the AI platform has already been developed through a previous IITM Transcript-Based RAG System.

This existing implementation serves as the centralized AI Knowledge Service that powers educational intelligence throughout the platform.

## **Existing Capabilities**

### **Knowledge Processing**

* Educational transcript ingestion  
* Transcript cleaning and preprocessing  
* Semantic chunking  
* Knowledge extraction  
* Embedding generation

### **Knowledge Base**

* Pinecone Vector Database  
* Semantic indexing  
* Metadata-based retrieval  
* Educational knowledge repository

### **AI Knowledge Engine**

* Retrieval-Augmented Generation (RAG)  
* Semantic search  
* Context-aware document retrieval  
* Context-aware question answering  
* Educational summarization  
* Intelligent content retrieval

### **AI Mentor**

* Functional AI Mentor chatbot  
* Educational conversations grounded in IITM content  
* Retrieval-powered responses using the existing RAG pipeline

The AI Mentor chatbot and RAG pipeline are already operational and will be reused throughout this project.

---

# **Scope of this Project (New Development)**

The focus of this project is to build the application layer around the existing AI Knowledge Service.

The following components will be developed:

* User Authentication  
* Learner Profile Management  
* AI Readiness Assessment  
* Personalized Learning Roadmap  
* AI Mentor Dashboard  
* Frontend User Interface  
* Backend APIs  
* Database Integration

The existing RAG pipeline and AI Mentor chatbot will remain the shared knowledge engine powering multiple platform features.

---

# **User Journey**

## **Step 1: Authentication**

Users securely sign in using Google Authentication or Supabase Authentication.

---

## **Step 2: Profile Creation**

After authentication, users complete a professional profile by providing:

* Current Role  
* Industry  
* Years of Experience  
* Career Aspirations  
* AI Learning Goals  
* Weekly Learning Availability

This profile is stored and used to personalize the learning experience.

---

## **Step 3: AI Mentor Dashboard**

After completing onboarding, users are taken to their personalized dashboard.

The dashboard serves as the central workspace where users can:

* View their learner profile  
* Access the AI Mentor  
* Explore available learning tracks  
* Begin the AI Readiness Assessment

A prominent **Start Assessment** button guides first-time users to begin their learning journey.

---

## **Step 4: AI Readiness Assessment**

Users complete an AI readiness assessment designed to evaluate their current AI knowledge and identify learning gaps.

Assessment areas include:

* AI Fundamentals  
* Generative AI  
* Prompt Engineering  
* AI Productivity  
* Business Applications of AI  
* Technical AI Concepts (where applicable)

Assessment results are stored and used to personalize learning recommendations.

---

## **Step 5: Personalized Learning Roadmap**

Based on:

* User Profile  
* Career Goals  
* Assessment Results  
* Existing IITM RAG Knowledge Base

the platform generates a personalized learning roadmap.

The learner is recommended one of three learning pathways.

### **AI Foundations Track**

Designed for professionals with limited technical exposure.

Example roles:

* HR  
* Finance  
* Operations  
* Administration  
* Customer Support

---

### **AI Practitioner Track**

Designed for professionals responsible for managing teams, projects, or business functions.

Example roles:

* Project Managers  
* Sales Professionals  
* Marketing Professionals  
* Business Analysts  
* Product Owners

---

### **AI Builder Track**

Designed for technical professionals building AI-enabled products and systems.

Example roles:

* Software Developers  
* Product Managers  
* Solutions Architects  
* Data Scientists  
* AI Engineers

---

Each roadmap includes:

* Recommended IITM learning modules  
* Recommended sessions  
* Suggested learning sequence  
* Weekly milestones  
* Estimated completion timeline  
* Skill-gap analysis

---

## **Step 6: AI Mentor Guided Learning**

Learners interact with the AI Mentor throughout their learning journey.

The AI Mentor leverages the existing centralized IITM RAG Knowledge Service to provide:

* Context-aware question answering  
* Concept explanations  
* Educational summaries  
* Lecture clarification  
* Topic recommendations  
* Learning guidance

Both the AI Mentor and Personalized Learning Roadmap utilize the same centralized RAG infrastructure.

---

# **System Architecture**

                   Streamlit Frontend  
                            │  
                    FastAPI Backend  
                            │  
        ┌─────────────────────────────────────┐  
        │        Application Services         │  
        │                                     │  
        │ • User Profile Service              │  
        │ • Assessment Engine                 │  
        │ • Learning Roadmap Generator        │  
        │ • Dashboard                         │  
        │ • AI Mentor Interface               │  
        └─────────────────────────────────────┘  
                            │  
              Shared AI Knowledge Service  
                    (Already Built)  
                            │  
          IITM Transcript-Based RAG Pipeline  
                            │  
          Pinecone Vector Database \+ LLM

---

# **Technical Stack**

### **Frontend**

* Streamlit

### **Backend**

* FastAPI

### **Authentication**

* Google Login / Supabase Auth

### **Database**

* Supabase

Stores:

* User Profiles  
* Assessment Results  
* Learning Roadmaps

### **AI Knowledge Layer (Existing)**

* IITM Transcript-Based RAG  
* Pinecone Vector Database  
* Context-Aware Retrieval  
* AI Mentor Chatbot

### **AI Application Layer (To Be Built)**

* Assessment Engine  
* Personalized Learning Roadmap Generator  
* User Dashboard  
* AI Mentor Integration

---

# **Success Metrics**

* User successfully completes onboarding  
* User completes AI readiness assessment  
* Personalized learning roadmap is generated  
* AI Mentor successfully answers educational queries using the existing RAG system  
* User can navigate the complete learning workflow from onboarding through guided learning

---

# **Future Scope (Out of Scope for MVP)**

The following features are intentionally excluded from the current implementation:

* Continuous Progress Tracking  
* Adaptive Roadmap Updates  
* Periodic Reassessments  
* Learning Analytics Dashboard  
* Gamification and Achievement Badges  
* Notifications and Learning Reminders  
* Long-term Personalized Recommendation Engine

---

# **Project Goal**

The objective of this project is **not** to build a new Retrieval-Augmented Generation (RAG) system.

Instead, the goal is to transform an existing IITM Transcript-Based RAG implementation into a complete AI Learning Mentor platform by building the application layer that enables personalized onboarding, AI readiness assessment, structured learning pathways, and an integrated AI Mentor experience.

The existing RAG infrastructure will function as a shared AI Knowledge Service, powering both the AI Mentor and the Personalized Learning Roadmap while ensuring all AI-generated educational guidance remains grounded in the IITM knowledge base.

