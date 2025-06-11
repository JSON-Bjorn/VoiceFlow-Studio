"""
Voice Name Resolver Service

This service handles dynamic resolution of host names from voice profiles and user inputs,
providing a clean interface for mapping between generic host identifiers and actual voice names.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class VoiceNameResolver:
    """
    Resolves dynamic host names from voice profiles and user configurations.

    This service provides the bridge between generic host identifiers (host_1, host_2)
    and actual voice names (David Professional, Marcus Conversational) for use throughout
    the podcast generation pipeline.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VoiceNameResolver")

    def resolve_host_names(
        self,
        user_inputs: Dict[str, Any],
        voice_profiles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Resolve actual host names from user inputs or voice profiles

        Args:
            user_inputs: User configuration including potential host overrides
            voice_profiles: Available voice profiles with names

        Returns:
            Dict mapping host IDs to resolved names
            Example: {"host_1": "David Professional", "host_2": "Marcus Conversational"}
        """
        try:
            resolved_names = {}

            # Get user-specified hosts configuration
            user_hosts = user_inputs.get("hosts", {})

            # Priority 1: Use user-specified host names if provided
            for host_id in ["host_1", "host_2"]:
                if host_id in user_hosts and "name" in user_hosts[host_id]:
                    resolved_names[host_id] = user_hosts[host_id]["name"]
                    self.logger.debug(
                        f"Using user-specified name for {host_id}: {resolved_names[host_id]}"
                    )

            # Priority 2: Use voice profile names for unspecified hosts
            if voice_profiles:
                for host_id in ["host_1", "host_2"]:
                    if host_id not in resolved_names and host_id in voice_profiles:
                        voice_name = voice_profiles[host_id].get(
                            "name", f"Host {host_id[-1]}"
                        )
                        resolved_names[host_id] = voice_name
                        self.logger.debug(
                            f"Using voice profile name for {host_id}: {voice_name}"
                        )

            # Priority 3: Fallback to generic names
            for host_id in ["host_1", "host_2"]:
                if host_id not in resolved_names:
                    fallback_name = f"Host {host_id[-1]}"
                    resolved_names[host_id] = fallback_name
                    self.logger.debug(
                        f"Using fallback name for {host_id}: {fallback_name}"
                    )

            self.logger.info(f"Resolved host names: {resolved_names}")
            return resolved_names

        except Exception as e:
            self.logger.error(f"Failed to resolve host names: {e}")
            # Return safe fallbacks
            return {"host_1": "Host 1", "host_2": "Host 2"}

    def get_speaker_mapping(
        self,
        user_inputs: Dict[str, Any],
        voice_profiles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """
        Get mapping from generic IDs to actual names for speaker identification

        Args:
            user_inputs: User configuration
            voice_profiles: Available voice profiles

        Returns:
            Dict mapping host IDs to short display names
            Example: {"host_1": "David", "host_2": "Marcus"}
        """
        try:
            resolved_names = self.resolve_host_names(user_inputs, voice_profiles)

            # Extract first names or create short versions
            speaker_mapping = {}
            for host_id, full_name in resolved_names.items():
                # Extract first name (before space) or use full name if short
                if " " in full_name:
                    short_name = full_name.split(" ")[0]
                else:
                    short_name = full_name

                speaker_mapping[host_id] = short_name

            self.logger.debug(f"Speaker mapping: {speaker_mapping}")
            return speaker_mapping

        except Exception as e:
            self.logger.error(f"Failed to create speaker mapping: {e}")
            return {"host_1": "Host1", "host_2": "Host2"}

    def resolve_voice_assignments(
        self,
        user_inputs: Dict[str, Any],
        voice_profiles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Resolve complete voice assignments including names and voice IDs

        Args:
            user_inputs: User configuration
            voice_profiles: Available voice profiles

        Returns:
            Complete voice assignment configuration
        """
        try:
            resolved_assignments = {}
            user_hosts = user_inputs.get("hosts", {})

            for host_id in ["host_1", "host_2"]:
                assignment = {
                    "host_id": host_id,
                    "name": "Host 1" if host_id == "host_1" else "Host 2",
                    "voice_id": None,
                    "source": "fallback",
                }

                # Check user-specified configuration first
                if host_id in user_hosts:
                    user_host = user_hosts[host_id]
                    if "name" in user_host:
                        assignment["name"] = user_host["name"]
                        assignment["source"] = "user_specified"
                    if "voice_id" in user_host:
                        assignment["voice_id"] = user_host["voice_id"]

                # Use voice profile data if available and not overridden
                if voice_profiles and host_id in voice_profiles:
                    voice_profile = voice_profiles[host_id]
                    if assignment["source"] != "user_specified":
                        assignment["name"] = voice_profile.get(
                            "name", assignment["name"]
                        )
                        assignment["source"] = "voice_profile"
                    if not assignment["voice_id"]:
                        assignment["voice_id"] = voice_profile.get("voice_id")

                resolved_assignments[host_id] = assignment

            self.logger.info(f"Resolved voice assignments: {resolved_assignments}")
            return resolved_assignments

        except Exception as e:
            self.logger.error(f"Failed to resolve voice assignments: {e}")
            return {
                "host_1": {
                    "host_id": "host_1",
                    "name": "Host 1",
                    "voice_id": None,
                    "source": "error_fallback",
                },
                "host_2": {
                    "host_id": "host_2",
                    "name": "Host 2",
                    "voice_id": None,
                    "source": "error_fallback",
                },
            }

    def validate_host_configuration(
        self, user_inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that host configuration is complete and valid

        Args:
            user_inputs: User configuration to validate

        Returns:
            Validation result with issues and recommendations
        """
        try:
            validation_result = {
                "valid": True,
                "issues": [],
                "warnings": [],
                "recommendations": [],
                "host_count": 0,
            }

            user_hosts = user_inputs.get("hosts", {})
            validation_result["host_count"] = len(user_hosts)

            # Check for required host IDs
            required_hosts = ["host_1", "host_2"]
            for host_id in required_hosts:
                if host_id not in user_hosts:
                    validation_result["warnings"].append(
                        f"Missing {host_id} configuration, will use defaults"
                    )
                else:
                    host_config = user_hosts[host_id]

                    # Validate host configuration structure
                    if not isinstance(host_config, dict):
                        validation_result["issues"].append(
                            f"{host_id} configuration must be a dictionary"
                        )
                        validation_result["valid"] = False
                        continue

                    # Check for name field
                    if "name" not in host_config or not host_config["name"].strip():
                        validation_result["warnings"].append(
                            f"{host_id} missing name, will use voice profile default"
                        )

                    # Check for voice_id field
                    if "voice_id" not in host_config or not host_config["voice_id"]:
                        validation_result["warnings"].append(
                            f"{host_id} missing voice_id, will use system default"
                        )

            # Add recommendations based on findings
            if validation_result["host_count"] == 0:
                validation_result["recommendations"].append(
                    "Consider specifying custom host names for personalized podcast experience"
                )

            if validation_result["warnings"]:
                validation_result["recommendations"].append(
                    "Provide complete host configurations (name + voice_id) for best results"
                )

            self.logger.debug(f"Host configuration validation: {validation_result}")
            return validation_result

        except Exception as e:
            self.logger.error(f"Failed to validate host configuration: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "recommendations": ["Check host configuration format"],
                "host_count": 0,
            }

    def create_enhanced_host_config(
        self,
        user_inputs: Dict[str, Any],
        voice_profiles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create enhanced host configuration merging user inputs with voice profiles

        Args:
            user_inputs: User configuration
            voice_profiles: Available voice profiles

        Returns:
            Enhanced host configuration ready for script generation
        """
        try:
            resolved_assignments = self.resolve_voice_assignments(
                user_inputs, voice_profiles
            )

            enhanced_config = {}

            for host_id, assignment in resolved_assignments.items():
                enhanced_config[host_id] = {
                    "name": assignment["name"],
                    "voice_id": assignment["voice_id"],
                    "personality": self._get_default_personality(host_id),
                    "role": self._get_default_role(host_id),
                    "speaking_style": self._get_default_speaking_style(host_id),
                    "source": assignment["source"],
                }

                # Merge any additional user-specified properties
                user_hosts = user_inputs.get("hosts", {})
                if host_id in user_hosts:
                    user_config = user_hosts[host_id]
                    for key, value in user_config.items():
                        if key not in [
                            "name",
                            "voice_id",
                        ]:  # Don't override resolved values
                            enhanced_config[host_id][key] = value

            self.logger.info(
                f"Created enhanced host config: {list(enhanced_config.keys())}"
            )
            return enhanced_config

        except Exception as e:
            self.logger.error(f"Failed to create enhanced host config: {e}")
            # Return basic fallback configuration
            return {
                "host_1": {
                    "name": "Host 1",
                    "personality": "analytical and engaging host",
                    "role": "primary_questioner",
                    "speaking_style": "thoughtful, probing questions",
                },
                "host_2": {
                    "name": "Host 2",
                    "personality": "warm and curious host",
                    "role": "storyteller",
                    "speaking_style": "energetic, relatable examples",
                },
            }

    def _get_default_personality(self, host_id: str) -> str:
        """Get default personality for a host ID"""
        personalities = {
            "host_1": "analytical and engaging host",
            "host_2": "warm and curious host",
        }
        return personalities.get(host_id, "professional podcast host")

    def _get_default_role(self, host_id: str) -> str:
        """Get default role for a host ID"""
        roles = {"host_1": "primary_questioner", "host_2": "storyteller"}
        return roles.get(host_id, "co_host")

    def _get_default_speaking_style(self, host_id: str) -> str:
        """Get default speaking style for a host ID"""
        styles = {
            "host_1": "thoughtful, probing questions",
            "host_2": "energetic, relatable examples",
        }
        return styles.get(host_id, "natural conversation style")
