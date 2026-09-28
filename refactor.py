import sys
import re

file_path = "api_server.py"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    '"alerts"': '"social_media_alerts"',
    '"contents"': '"social_media_posts"',
    '"events"': '"social_media_events"',
    '"pois"': '"social_media_profiles"',
    '"sources"': '"social_media_accounts"',
    '"grievances"': '"social_media_grievances"',
    '"grievance_workflow_reports"': '"social_media_grievance_reports"',
    '"dial100incidents"': '"dial100incidents"', # Keep as is if not in db
    "'alerts'": "'social_media_alerts'",
    "'contents'": "'social_media_posts'",
    "'events'": "'social_media_events'",
    "'pois'": "'social_media_profiles'",
    "'sources'": "'social_media_accounts'",
    "'grievances'": "'social_media_grievances'",
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)

print("api_server.py has been updated successfully!")
