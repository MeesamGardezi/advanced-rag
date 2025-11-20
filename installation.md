# Installation Guide - Agentic RAG System

## Prerequisites

- Python 3.9 or higher
- PostgreSQL 14 or higher installed and running
- Firebase project with Firestore database
- OpenAI API key

## Step 1: Install PostgreSQL

### macOS (using Homebrew)
```bash
brew install postgresql@14
brew services start postgresql@14
```

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### Windows
Download and install from: https://www.postgresql.org/download/windows/

## Step 2: Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE construction_rag;

# Create user (optional, for production)
CREATE USER rag_user WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE construction_rag TO rag_user;

# Exit
\q
```

## Step 3: Install Python Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements_new.txt
```

## Step 4: Configure Environment Variables

1. Copy `.env.example` to `.env` (or update existing `.env`)
2. Fill in all required values:
   - PostgreSQL connection details
   - OpenAI API key
   - Firebase credentials

### Required Environment Variables

```bash
# PostgreSQL
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=construction_rag
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Firebase (use your existing credentials)
FIREBASE_PROJECT_ID=your_project_id
FIREBASE_PRIVATE_KEY_ID=your_key_id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your_service_account@project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your_client_id
```

## Step 5: Initialize Database Schema

```bash
# Test connection and create tables
python db_connection.py
```

This will:
- Test PostgreSQL connection
- Create all tables (jobs, estimates, flooring_estimates, schedule_items, consumed_items)
- Show connection pool status

Expected output:
```
🧪 Testing database connection...
✅ Database initialization successful
📊 Pool status: {...}
📊 Table row counts: {...}
✅ Database connection closed
```

## Step 6: Migrate Data (Single Job)

```bash
# Migrate a specific job from Firestore to PostgreSQL
python etl_migration.py --company-id YOUR_COMPANY_ID --job-id YOUR_JOB_ID
```

Example:
```bash
python etl_migration.py --company-id comp123 --job-id job456
```

To overwrite existing data:
```bash
python etl_migration.py --company-id comp123 --job-id job456 --force
```

Expected output:
```
================================================================================
🚀 Starting ETL migration for job: comp123/job456
================================================================================
📥 Fetching job data from Firestore...
✅ Job data fetched: Hammond 2508
🔄 Starting database transaction...
📝 Migrating job metadata...
✅ Job metadata: Hammond 2508
📝 Migrating estimate data (45 rows)...
✅ Estimates: 45 rows
💾 Committing transaction...
🔍 Verifying migration...
✅ Verification successful - all counts match!
================================================================================
✅ Migration completed in 3.45 seconds
================================================================================
```

## Step 7: Verify Migration

### Option 1: Python Script
```python
from db_connection import get_job_data_summary

summary = get_job_data_summary('your_job_id')
print(summary)
```

### Option 2: PostgreSQL CLI
```bash
psql -U postgres -d construction_rag -c "
  SELECT 
    j.name as job_name,
    (SELECT COUNT(*) FROM estimates WHERE job_id = j.id) as estimates,
    (SELECT COUNT(*) FROM flooring_estimates WHERE job_id = j.id) as flooring,
    (SELECT COUNT(*) FROM schedule_items WHERE job_id = j.id) as schedule,
    (SELECT COUNT(*) FROM consumed_items WHERE job_id = j.id) as consumed
  FROM jobs j
  WHERE j.id = 'your_job_id';
"
```

## Troubleshooting

### PostgreSQL Connection Issues

**Error:** "could not connect to server"

**Solution:** Ensure PostgreSQL is running:
```bash
# Check status
brew services list  # macOS
sudo systemctl status postgresql  # Linux

# Restart if needed
brew services restart postgresql@14  # macOS
sudo systemctl restart postgresql  # Linux
```

**Error:** "password authentication failed"

**Solution:** Check your `.env` file has correct credentials. Also verify PostgreSQL accepts password authentication:
```bash
# Edit pg_hba.conf (location varies by OS)
# Change 'peer' or 'ident' to 'md5' for local connections
# macOS: /usr/local/var/postgresql@14/pg_hba.conf
# Linux: /etc/postgresql/14/main/pg_hba.conf
```

**Error:** "role 'postgres' does not exist"

**Solution:** Create the postgres user:
```bash
createuser -s postgres
```

### Migration Issues

**Error:** "Job already exists"

**Solution:** Use `--force` flag or delete existing data:
```python
from db_connection import session_scope
from database_schema import Job

with session_scope() as session:
    job = session.query(Job).filter(Job.id == 'your_job_id').first()
    if job:
        session.delete(job)
        # Cascade delete will remove all related records
```

**Error:** "Firebase connection failed"

**Solution:** 
1. Verify Firebase credentials in `.env`
2. Ensure FIREBASE_PRIVATE_KEY has proper newline formatting: `\n`
3. Test Firebase connection separately:
```python
from database import initialize_firebase
initialize_firebase()
```

**Error:** "Module not found: langchain"

**Solution:** Ensure you installed the new requirements:
```bash
pip install -r requirements_new.txt
```

### Data Verification Issues

**Error:** "Count mismatch after migration"

**Solution:** 
1. Check migration logs for warnings
2. Verify source data in Firestore
3. Re-run migration with `--force` flag
4. Check for empty/invalid entries that may be filtered out

## Next Steps

After successful installation and data migration:

1. ✅ Phase 1 Complete: Database schema and ETL
2. 🔄 Phase 2: Tool Development (SQL Tool + Vector Tool)
3. 🔄 Phase 3: Agent System (LangChain configuration)
4. 🔄 Phase 4: API Layer (FastAPI endpoints)

## Useful Commands

### Database Management

```bash
# Connect to database
psql -U postgres -d construction_rag

# List all tables
\dt

# Describe table structure
\d estimates

# View row counts for all tables
SELECT 
  'jobs' as table_name, COUNT(*) as count FROM jobs
UNION ALL
SELECT 'estimates', COUNT(*) FROM estimates
UNION ALL
SELECT 'flooring_estimates', COUNT(*) FROM flooring_estimates
UNION ALL
SELECT 'schedule_items', COUNT(*) FROM schedule_items
UNION ALL
SELECT 'consumed_items', COUNT(*) FROM consumed_items;

# Exit
\q
```

### Development Commands

```bash
# Test database connection
python db_connection.py

# Run migration with verbose output
python etl_migration.py --company-id X --job-id Y

# Check table counts programmatically
python -c "from db_connection import get_table_row_counts; print(get_table_row_counts())"

# Health check
python -c "from db_connection import health_check; import json; print(json.dumps(health_check(), indent=2))"
```

### Clean Up (Caution!)

```bash
# Drop all tables and recreate
python -c "
from database_schema import drop_all_tables, create_all_tables
from db_connection import get_db
db = get_db()
drop_all_tables(db.engine)
create_all_tables(db.engine)
"

# Delete specific job
python -c "
from db_connection import session_scope
from database_schema import Job
with session_scope() as session:
    job = session.query(Job).filter(Job.id == 'YOUR_JOB_ID').first()
    if job:
        session.delete(job)
        print(f'Deleted job: {job.name}')
"
```

## Performance Tips

1. **Indexes**: The schema includes strategic indexes. Monitor query performance:
   ```sql
   EXPLAIN ANALYZE SELECT * FROM estimates WHERE cost_code = '01-5020';
   ```

2. **Connection Pooling**: Adjust pool settings in `.env` based on load:
   ```bash
   POSTGRES_POOL_SIZE=20
   POSTGRES_MAX_OVERFLOW=10
   ```

3. **Query Optimization**: Use `EXPLAIN` to understand query plans
4. **Regular Maintenance**: Run `VACUUM ANALYZE` periodically

## Security Considerations

1. **Never commit `.env`**: Add to `.gitignore`
2. **Use strong passwords**: For production PostgreSQL
3. **Restrict access**: Configure PostgreSQL `pg_hba.conf` appropriately
4. **SSL connections**: For production, enable SSL in PostgreSQL
5. **Backup regularly**: Set up automated PostgreSQL backups

## Support & Resources

- PostgreSQL Documentation: https://www.postgresql.org/docs/
- SQLAlchemy Documentation: https://docs.sqlalchemy.org/
- LangChain Documentation: https://python.langchain.com/docs/
- Project Documentation: See `docs/` folder (coming soon)

## Checklist

Before proceeding to Phase 2, ensure:

- [ ] PostgreSQL is installed and running
- [ ] Database `construction_rag` is created
- [ ] Python dependencies are installed
- [ ] `.env` file is configured with all required variables
- [ ] Database schema is created (tables exist)
- [ ] At least one job is successfully migrated
- [ ] Migration verification passed (counts match)
- [ ] Health check returns "healthy" status

If all items are checked, you're ready for Phase 2! 🚀