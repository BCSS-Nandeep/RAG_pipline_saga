import sys
import pipeline

TABLES_TO_INGEST = [
    "social_media_alerts",
    "social_media_posts",
    "social_media_events",
    "social_media_grievances",
    "social_media_grievance_reports",
    "social_media_profiles",
    "social_media_accounts",
    "keywords",
    "platforms"
]

def main():
    print("Starting full DB ingestion natively...")
    for table in TABLES_TO_INGEST:
        print(f"\\n{'='*60}\\nIngesting table: {table}\\n{'='*60}")
        # Override the collection name globally for the pipeline module
        pipeline.COLLECTION_NAME = table
        try:
            # Force full re-ingestion for each table
            pipeline.run_ingestion(full=True)
        except Exception as e:
            print(f"Error ingesting table {table}: {e}")

    print("\\nFull database ingestion process finished.")

if __name__ == "__main__":
    main()
