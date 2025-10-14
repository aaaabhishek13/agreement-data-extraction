"""
Pydantic models for lease agreement data structure and validation.
"""

from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field, validator


class Location(BaseModel):
    """Location details of the property."""
    survey_block_plot_no: Optional[str] = None
    district_sector: Optional[str] = None
    village: Optional[str] = None
    zone_khewat: Optional[str] = None
    mandal_municipality_khata: Optional[str] = None
    registrar_office_id: Optional[str] = None


class AgreementDetails(BaseModel):
    """Basic agreement information."""
    serviced_office: Optional[bool] = None
    document_number: Optional[str] = None
    project_name: Optional[str] = None
    location: Optional[Location] = None
    agreement_type: Optional[str] = None
    agreement_date: Optional[date] = None


class PartyInfo(BaseModel):
    """Information about a party (landlord or tenant)."""
    name: Optional[str] = None
    representative_name: Optional[str] = None
    representative_role: Optional[str] = None


class Parties(BaseModel):
    """Landlord and tenant information."""
    landlord: Optional[PartyInfo] = None
    tenant: Optional[PartyInfo] = None


class UnitDetails(BaseModel):
    """Details about the leased unit."""
    unit_number: Optional[str] = None
    floor_number: Optional[str] = None
    wing: Optional[str] = None
    other_info: Optional[str] = None
    abstract_area_type: Optional[str] = None
    abstract_area: Optional[float] = Field(None, ge=0)
    abstract_rate: Optional[float] = Field(None, ge=0)
    chargeable_area_type: Optional[str] = None
    super_built_up_area: Optional[float] = Field(None, ge=0)
    built_up_area: Optional[float] = Field(None, ge=0)
    carpet_area: Optional[float] = Field(None, ge=0)


class Escalation(BaseModel):
    """Rent escalation details."""
    period_months: Optional[int] = Field(None, ge=0)
    percentage: Optional[float] = Field(None, ge=0)


class LeaseTerms(BaseModel):
    """Lease terms and conditions."""
    lease_start_date: Optional[date] = None
    lease_expiry_date: Optional[date] = None
    license_duration_months: Optional[int] = Field(None, ge=0)
    monthly_rent: Optional[float] = Field(None, ge=0)
    rate_per_sqft: Optional[float] = Field(None, ge=0)
    consideration_value: Optional[float] = Field(None, ge=0)
    escalation: Optional[Escalation] = None
    unit_condition: Optional[str] = None
    fit_outs: Optional[bool] = None
    furnished_rate: Optional[float] = Field(None, ge=0)
    rent_free_period_months: Optional[int] = Field(None, ge=0)
    lock_in_period_landlord_months: Optional[int] = Field(None, ge=0)
    lock_in_period_tenant_months: Optional[int] = Field(None, ge=0)
    
    @validator('lease_expiry_date')
    def validate_lease_dates(cls, v, values):
        """Ensure lease expiry date is after start date."""
        if v and values.get('lease_start_date') and v <= values['lease_start_date']:
            raise ValueError('Lease expiry date must be after start date')
        return v


class Financials(BaseModel):
    """Financial information."""
    security_deposit: Optional[float] = Field(None, ge=0)
    monthly_rental_equivalent_of_deposit: Optional[float] = Field(None, ge=0)
    market_value: Optional[float] = Field(None, ge=0)
    stamp_duty_amount: Optional[float] = Field(None, ge=0)
    registration_amount: Optional[float] = Field(None, ge=0)


class ParkingCAM(BaseModel):
    """Parking and Common Area Maintenance charges."""
    car_parking_slots: Optional[int] = Field(None, ge=0)
    car_parking_type: Optional[str] = None
    additional_car_parking_charges: Optional[float] = Field(None, ge=0)
    two_wheeler_parking_slots: Optional[int] = Field(None, ge=0)
    additional_two_wheeler_parking_charges: Optional[float] = Field(None, ge=0)
    monthly_cam_charges: Optional[float] = Field(None, ge=0)
    cam_paid_by: Optional[str] = None


class PropertyTax(BaseModel):
    """Property tax information."""
    total_property_tax: Optional[float] = Field(None, ge=0)
    property_tax: Optional[float] = Field(None, ge=0)
    paid_by: Optional[str] = None


class Miscellaneous(BaseModel):
    """Miscellaneous information."""
    comments: Optional[str] = None
    approver_comments: Optional[str] = None
    floor_plan: Optional[str] = None
    abstract: Optional[str] = None
    agreement_file: Optional[str] = None
    other_documents: Optional[List[str]] = None


class Citations(BaseModel):
    """Citation information for extracted fields."""
    agreement_details: Optional[dict] = None
    parties: Optional[dict] = None
    unit_details: Optional[dict] = None
    lease_terms: Optional[dict] = None
    financials: Optional[dict] = None
    parking_cam: Optional[dict] = None
    property_tax: Optional[dict] = None
    miscellaneous: Optional[dict] = None


class LeaseAgreementData(BaseModel):
    """Complete lease agreement data structure."""
    agreement_details: Optional[AgreementDetails] = None
    parties: Optional[Parties] = None
    unit_details: Optional[UnitDetails] = None
    lease_terms: Optional[LeaseTerms] = None
    financials: Optional[Financials] = None
    parking_cam: Optional[ParkingCAM] = None
    property_tax: Optional[PropertyTax] = None
    miscellaneous: Optional[Miscellaneous] = None
    citations: Optional[Citations] = None
    
    # Metadata fields
    extraction_status: Optional[str] = None
    provider: Optional[str] = None
    error: Optional[str] = None
    token_usage: Optional[dict] = None
    extracted_text: Optional[str] = None  # Store original extracted text for viewer
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }
        
    def to_dict(self) -> dict:
        """Convert to dictionary with proper serialization."""
        return self.model_dump(exclude_none=False, by_alias=True)
    
    def summary(self) -> dict:
        """Get a summary of the extracted data."""
        summary_data = {}
        
        # Count populated fields
        total_fields = 0
        populated_fields = 0
        
        for section_name, section_data in self.model_dump().items():
            if section_name in ['extraction_status', 'provider', 'error']:
                continue
                
            if isinstance(section_data, dict):
                section_total, section_populated = self._count_fields(section_data)
                total_fields += section_total
                populated_fields += section_populated
                
                summary_data[section_name] = {
                    'total_fields': section_total,
                    'populated_fields': section_populated,
                    'completion_rate': round(section_populated / section_total * 100, 1) if section_total > 0 else 0
                }
        
        summary_data['overall'] = {
            'total_fields': total_fields,
            'populated_fields': populated_fields,
            'completion_rate': round(populated_fields / total_fields * 100, 1) if total_fields > 0 else 0,
            'extraction_status': self.extraction_status,
            'provider': self.provider
        }
        
        if self.error:
            summary_data['overall']['error'] = self.error
            
        return summary_data
    
    def _count_fields(self, data: dict) -> tuple[int, int]:
        """Count total and populated fields in a nested structure."""
        total = 0
        populated = 0
        
        for key, value in data.items():
            if isinstance(value, dict):
                nested_total, nested_populated = self._count_fields(value)
                total += nested_total
                populated += nested_populated
            else:
                total += 1
                if value is not None:
                    populated += 1
        
        return total, populated


class ExtractionResult(BaseModel):
    """Result of the extraction process."""
    success: bool
    data: Optional[LeaseAgreementData] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    file_info: Optional[dict] = None
    
    class Config:
        """Pydantic configuration."""
        json_encoders = {
            date: lambda v: v.isoformat() if v else None
        }
