"""
SQLAlchemy models for PostgreSQL database
Matches Firestore structure with proper relationships and indexes
"""

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Text, 
    Boolean, ForeignKey, JSON, Index, CheckConstraint
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class Job(Base):
    """Job/Project master table"""
    __tablename__ = 'jobs'
    
    # Primary identification
    id = Column(String(255), primary_key=True)  # Firebase document ID
    company_id = Column(String(255), nullable=False, index=True)
    
    # Job metadata
    name = Column(String(500), nullable=False)  # projectTitle
    client_name = Column(String(500))
    site_city = Column(String(255))
    site_state = Column(String(100))
    site_location = Column(String(500))  # Computed: city, state
    project_description = Column(Text)
    
    # Status and type
    status = Column(String(50), default='active')
    estimate_type = Column(String(50), default='general')
    
    # Timestamps
    created_date = Column(DateTime)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    schedule_last_updated = Column(DateTime)
    
    # Relationships
    estimates = relationship("Estimate", back_populates="job", cascade="all, delete-orphan")
    flooring_estimates = relationship("FlooringEstimate", back_populates="job", cascade="all, delete-orphan")
    schedule_items = relationship("ScheduleItem", back_populates="job", cascade="all, delete-orphan")
    consumed_items = relationship("ConsumedItem", back_populates="job", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_job_company_id', 'company_id'),
        Index('idx_job_name', 'name'),
        Index('idx_job_status', 'status'),
    )
    
    def __repr__(self):
        return f"<Job(id={self.id}, name={self.name}, client={self.client_name})>"


class Estimate(Base):
    """Estimate line items table"""
    __tablename__ = 'estimates'
    
    # Primary identification
    id = Column(String(255), primary_key=True)
    job_id = Column(String(255), ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    row_number = Column(Integer, nullable=False)
    
    # Categorization (for exact filtering)
    area = Column(String(500), index=True)
    task_scope = Column(String(500), index=True)
    cost_code = Column(String(255), index=True)
    row_type = Column(String(50), default='estimate')  # 'estimate' or 'allowance'
    
    # Text fields (for semantic search)
    description = Column(Text)
    notes_remarks = Column(Text)
    
    # Quantities and units
    units = Column(String(100))
    qty = Column(Float, default=0.0)
    
    # Costs - Estimated
    rate = Column(Float, default=0.0)
    total = Column(Float, nullable=False, default=0.0, index=True)
    
    # Costs - Budgeted
    budgeted_rate = Column(Float, default=0.0)
    budgeted_total = Column(Float, default=0.0, index=True)
    
    # Computed variance
    variance = Column(Float, default=0.0)  # total - budgeted_total
    variance_pct = Column(Float, default=0.0)  # (variance / budgeted_total) * 100
    
    # Materials (stored as JSON array)
    materials = Column(JSON)
    has_materials = Column(Boolean, default=False)
    material_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="estimates")
    
    # Constraints and indexes
    __table_args__ = (
        CheckConstraint("row_type IN ('estimate', 'allowance')", name='check_estimate_row_type'),
        Index('idx_estimate_job_id', 'job_id'),
        Index('idx_estimate_cost_code', 'cost_code'),
        Index('idx_estimate_area', 'area'),
        Index('idx_estimate_row_number', 'job_id', 'row_number'),
        Index('idx_estimate_total', 'total'),
        Index('idx_estimate_row_type', 'row_type'),
    )
    
    def __repr__(self):
        return f"<Estimate(id={self.id}, row={self.row_number}, cost_code={self.cost_code}, total=${self.total})>"


class FlooringEstimate(Base):
    """Flooring estimate line items table"""
    __tablename__ = 'flooring_estimates'
    
    # Primary identification
    id = Column(String(255), primary_key=True)
    job_id = Column(String(255), ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    row_number = Column(Integer, nullable=False)
    
    # Flooring specific fields
    floor_type_id = Column(String(255), index=True)
    vendor = Column(String(500), index=True)
    item_material_name = Column(String(500), index=True)
    brand = Column(String(255))
    unit = Column(String(100))
    
    # Quantities
    measured_qty = Column(Float, default=0.0)
    supplier_qty = Column(Float, default=0.0)
    waste_factor = Column(Float, default=0.0)  # Percentage
    qty_including_waste = Column(Float, default=0.0)
    
    # Costs
    unit_price = Column(Float, default=0.0)
    cost_price = Column(Float, default=0.0)
    tax_freight = Column(Float, default=0.0)
    total_cost = Column(Float, nullable=False, default=0.0, index=True)
    sale_price = Column(Float, default=0.0, index=True)
    
    # Computed profit
    profit = Column(Float, default=0.0)  # sale_price - total_cost
    margin_pct = Column(Float, default=0.0)  # (profit / sale_price) * 100
    
    # Text fields (for semantic search)
    notes_remarks = Column(Text)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="flooring_estimates")
    
    # Indexes
    __table_args__ = (
        Index('idx_flooring_job_id', 'job_id'),
        Index('idx_flooring_vendor', 'vendor'),
        Index('idx_flooring_floor_type', 'floor_type_id'),
        Index('idx_flooring_row_number', 'job_id', 'row_number'),
        Index('idx_flooring_total_cost', 'total_cost'),
    )
    
    def __repr__(self):
        return f"<FlooringEstimate(id={self.id}, row={self.row_number}, item={self.item_material_name}, sale=${self.sale_price})>"


class ScheduleItem(Base):
    """Schedule/timeline items table"""
    __tablename__ = 'schedule_items'
    
    # Primary identification
    id = Column(String(255), primary_key=True)
    job_id = Column(String(255), ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    row_number = Column(Integer, nullable=False)
    
    # Task details
    task = Column(Text, nullable=False)
    is_main_task = Column(Boolean, default=False, index=True)
    task_type = Column(String(100), default='labour')  # 'labour', 'material', etc.
    
    # Hours
    hours = Column(Float, default=0.0)
    consumed = Column(Float, default=0.0)  # Actual hours consumed
    
    # Progress
    percentage_complete = Column(Float, default=0.0)
    
    # Dates
    start_date = Column(DateTime)
    end_date = Column(DateTime)
    
    # Resources (stored as JSON object)
    resources = Column(JSON)  # {resource_name: hours}
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="schedule_items")
    
    # Indexes
    __table_args__ = (
        Index('idx_schedule_job_id', 'job_id'),
        Index('idx_schedule_main_task', 'is_main_task'),
        Index('idx_schedule_dates', 'start_date', 'end_date'),
        Index('idx_schedule_row_number', 'job_id', 'row_number'),
    )
    
    def __repr__(self):
        return f"<ScheduleItem(id={self.id}, task={self.task[:50]}, progress={self.percentage_complete}%)>"


class ConsumedItem(Base):
    """Consumed/actual cost items table"""
    __tablename__ = 'consumed_items'
    
    # Primary identification
    id = Column(String(255), primary_key=True)
    job_id = Column(String(255), ForeignKey('jobs.id', ondelete='CASCADE'), nullable=False)
    
    # Cost details
    cost_code = Column(String(255), nullable=False, index=True)
    amount = Column(Float, nullable=False, default=0.0, index=True)
    
    # Categorization (computed from cost_code)
    category = Column(String(100), index=True)  # Materials, Labor, Subcontractors, Other
    
    # Metadata
    last_updated = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    job = relationship("Job", back_populates="consumed_items")
    
    # Indexes
    __table_args__ = (
        Index('idx_consumed_job_id', 'job_id'),
        Index('idx_consumed_cost_code', 'cost_code'),
        Index('idx_consumed_category', 'category'),
        Index('idx_consumed_amount', 'amount'),
    )
    
    def __repr__(self):
        return f"<ConsumedItem(id={self.id}, cost_code={self.cost_code}, amount=${self.amount})>"


# Utility function to create all tables
def create_all_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(engine)
    print("✅ All tables created successfully")


# Utility function to drop all tables (use with caution!)
def drop_all_tables(engine):
    """Drop all tables from the database"""
    Base.metadata.drop_all(engine)
    print("🗑️  All tables dropped")