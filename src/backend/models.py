"""SQLAlchemy database models."""

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    ForeignKey,
    UniqueConstraint,
    func,
    text,
)

from src.backend.database import Base


class User(Base):
    """Application user associated with a hospital."""

    __tablename__ = "users"

    # Unique user ID.
    id = Column(
        Integer,
        primary_key=True,
        index=True,
    )

    # User's full name.
    full_name = Column(
        String(150),
        nullable=False,
    )

    # Hospital or healthcare organization.
    hospital_name = Column(
        String(255),
        nullable=False,
    )

    # Unique login email.
    email = Column(
        String(255),
        unique=True,
        index=True,
        nullable=False,
    )

    # Secure password hash, never the original password.
    hashed_password = Column(
        String(255),
        nullable=False,
    )

    # False = hospital user.
    # True = administrator.
    is_admin = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

    # Account activation status.
    is_active = Column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )

    # Account registration timestamp.
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    # Records acceptance of platform terms.
    terms_accepted = Column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
    )

class AuditLog(Base):
    """Non-sensitive account and platform events for the admin activity feed."""

    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False, index=True)
    description = Column(String(500), nullable=False)
    actor_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        index=True,
    )


class HospitalMembership(Base):
    """Admin-approved link between a login and its hospital inventory record."""

    __tablename__ = "hospital_memberships"

    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    hospital_name = Column(String(255), nullable=False, index=True)
    approved_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class HospitalOperationalRecord(Base):
    """Hospital-specific inventory; deliberately separate from ward CSV rows."""

    __tablename__ = "hospital_operational_records"

    id = Column(Integer, primary_key=True, index=True)
    hospital_name = Column(String(255), nullable=False, unique=True, index=True)
    actual_available_beds = Column(Integer, nullable=True)
    saline_stock_available = Column(Integer, nullable=True)
    diagnostic_kits_stock_available = Column(Integer, nullable=True)
    updated_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class HospitalWardAssignment(Base):
    """Admin-controlled hospital-to-ward association used only for aggregation."""

    __tablename__ = "hospital_ward_assignments"
    __table_args__ = (UniqueConstraint("hospital_name", "ward_id", name="uq_hospital_ward"),)

    id = Column(Integer, primary_key=True, index=True)
    hospital_name = Column(String(255), nullable=False, index=True)
    ward_id = Column(String(100), nullable=False, index=True)
    city = Column(String(100), nullable=False)
    assigned_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class OperationsUserScope(Base):
    """Explicit city or ward scope granted to logistics users by an admin."""

    __tablename__ = "operations_user_scopes"
    __table_args__ = (UniqueConstraint("user_id", "city", "ward_id", name="uq_operations_scope"),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    city = Column(String(100), nullable=False)
    ward_id = Column(String(100), nullable=True)
    granted_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class WardOperationsRecord(Base):
    """Dynamic ward crew availability kept apart from the stable reference CSV."""

    __tablename__ = "ward_operations_records"

    ward_id = Column(String(100), primary_key=True)
    crew_count_available = Column(Integer, nullable=True)
    updated_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
