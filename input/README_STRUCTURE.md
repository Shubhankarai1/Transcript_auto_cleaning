Use this structure for transcript ingestion.

Rules:
- Put raw transcript text into the relevant `session_<number>.txt` file.
- Keep one transcript session per file.
- Create `session_2.txt`, `session_3.txt`, and so on in the same module folder as needed.
- Use lowercase folder names with underscores for module folders.
- Advanced keeps the existing module structure: `cms`, `map`, `wdp`.

Structure:

input/
  level_1_foundations/
    common_modules/
      <module_name>/
        session_1.txt
    subject_matter_expertise/
      <module_name>/
        session_1.txt
  level_2_intermediate/
    common_modules/
      <module_name>/
        session_1.txt
    role_specific/
      <module_name>/
        session_1.txt
  level_3_advanced/
    cms/
      session_1.txt
    map/
      session_2.txt
    wdp/
      session_1.txt
