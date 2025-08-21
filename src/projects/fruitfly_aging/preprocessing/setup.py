"""
Aging-specific data setup wrapper

Extends the shared DataSetupManager with aging-specific functionality.
For now, this just provides a compatibility alias - aging-specific features can be added later.
"""

from shared.data.setup import DataSetupManager as BaseDataSetupManager
import logging

logger = logging.getLogger(__name__)


class AgingDataSetupManager(BaseDataSetupManager):
    """
    Aging-specific data setup manager.

    Currently inherits all functionality from the base DataSetupManager.
    Can be extended with aging-specific setup logic in the future.
    """

    def __init__(self, config=None):
        """Initialize with aging-specific defaults if needed."""
        super().__init__(config)

    def setup_aging_data(self):
        """
        Aging-specific data setup workflow.

        Could include aging-specific validation, data quality checks, etc.
        For now, just calls the standard setup.
        """
        logger.info("Setting up aging research data...")
        return self.setup_data()


# Compatibility alias for existing code
DataSetupManager = AgingDataSetupManager
