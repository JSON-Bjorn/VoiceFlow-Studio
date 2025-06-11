"""
Voice Assignment Validator Service

This service validates that voice assignments are working correctly throughout the
podcast generation pipeline, providing early detection of assignment failures
and detailed diagnostics for troubleshooting.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)


class VoiceAssignmentValidator:
    """
    Validates that voice assignments are working correctly and provides
    comprehensive diagnostics for voice assignment issues.

    This service acts as a quality gate to prevent silent voice assignment
    failures and ensures both hosts get properly assigned voices.
    """

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VoiceAssignmentValidator")

    def validate_speaker_mapping(
        self, script_segments: List[Dict[str, Any]], voice_profiles: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that all speakers in script can be mapped to voices

        Args:
            script_segments: List of script segments with speaker information
            voice_profiles: Available voice profiles for mapping

        Returns:
            {
                "valid": True/False,
                "missing_mappings": [],
                "speaker_distribution": {"host_1": 45, "host_2": 55},
                "issues": [],
                "warnings": [],
                "recommendations": []
            }
        """
        try:
            validation_result = {
                "valid": True,
                "missing_mappings": [],
                "speaker_distribution": {},
                "voice_coverage": {},
                "issues": [],
                "warnings": [],
                "recommendations": [],
                "total_segments": len(script_segments),
                "validation_timestamp": time.time(),
                "debug_info": {},  # Add debug information
            }

            if not script_segments:
                validation_result["valid"] = False
                validation_result["issues"].append(
                    "No script segments provided for validation"
                )
                return validation_result

            if not voice_profiles:
                validation_result["valid"] = False
                validation_result["issues"].append(
                    "No voice profiles available for mapping"
                )
                return validation_result

            # Analyze speaker distribution and voice mapping
            speaker_counts = {}
            speaker_voice_mapping = {}
            unmapped_speakers = set()
            total_segments = len(script_segments)

            # Debug: collect all unique speakers found
            all_speakers_found = set()

            for segment in script_segments:
                speaker = segment.get("speaker", "unknown")
                voice_id = segment.get("voice_id")

                all_speakers_found.add(speaker)

                # Count speaker occurrences
                speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1

                # Track voice mapping
                if voice_id:
                    speaker_voice_mapping[speaker] = voice_id
                elif speaker in voice_profiles:
                    # Speaker exists in profiles but no voice_id in segment
                    speaker_voice_mapping[speaker] = voice_profiles[speaker].get(
                        "voice_id"
                    )
                else:
                    # Speaker not found in voice profiles
                    unmapped_speakers.add(speaker)

            # Add debug information
            validation_result["debug_info"] = {
                "all_speakers_found": list(all_speakers_found),
                "available_voice_profiles": list(voice_profiles.keys()),
                "speaker_counts": speaker_counts,
                "speaker_voice_mapping": speaker_voice_mapping,
                "unmapped_speakers": list(unmapped_speakers),
            }

            # Calculate speaker distribution percentages
            if total_segments > 0:
                for speaker, count in speaker_counts.items():
                    percentage = (count / total_segments) * 100
                    validation_result["speaker_distribution"][speaker] = {
                        "count": count,
                        "percentage": round(percentage, 1),
                    }

            # Analyze voice coverage
            available_voices = set(voice_profiles.keys())
            mapped_speakers = set(speaker_voice_mapping.keys())
            validation_result["voice_coverage"] = {
                "available_voices": list(available_voices),
                "mapped_speakers": list(mapped_speakers),
                "coverage_percentage": round(
                    (len(mapped_speakers) / len(available_voices)) * 100, 1
                )
                if available_voices
                else 0,
            }

            # Check for missing mappings - make this a warning, not a hard failure
            if unmapped_speakers:
                validation_result["missing_mappings"] = list(unmapped_speakers)
                validation_result["warnings"].append(
                    f"Speakers without voice mappings: {', '.join(unmapped_speakers)}"
                )
                # Only fail if ALL speakers are unmapped
                if len(unmapped_speakers) == len(all_speakers_found):
                    validation_result["valid"] = False
                    validation_result["issues"].append(
                        "All speakers are unmapped - voice generation will fail"
                    )

            # Validate speaker balance
            balance_issues = self._validate_speaker_balance(
                validation_result["speaker_distribution"]
            )
            if balance_issues:
                validation_result["warnings"].extend(balance_issues)

            # Check for required hosts - be more flexible here
            required_hosts = ["host_1", "host_2"]
            missing_hosts = []
            found_hosts = []
            for host in required_hosts:
                if host not in speaker_counts:
                    missing_hosts.append(host)
                else:
                    found_hosts.append(host)

            if missing_hosts and not found_hosts:
                # Only fail if NO required hosts are found
                validation_result["issues"].append(
                    f"No required hosts found - expected at least one of: {', '.join(required_hosts)}"
                )
                validation_result["valid"] = False
            elif missing_hosts:
                # Just warn if some hosts are missing
                validation_result["warnings"].append(
                    f"Missing some expected hosts: {', '.join(missing_hosts)}"
                )

            # Generate recommendations
            recommendations = self._generate_mapping_recommendations(
                validation_result["speaker_distribution"],
                validation_result["missing_mappings"],
                voice_profiles,
            )
            validation_result["recommendations"].extend(recommendations)

            # Log validation summary with debug info
            if validation_result["valid"]:
                self.logger.info(
                    f"Voice mapping validation PASSED: {len(mapped_speakers)} speakers mapped successfully"
                )
                self.logger.debug(f"Debug info: {validation_result['debug_info']}")
            else:
                self.logger.warning(
                    f"Voice mapping validation FAILED: {len(validation_result['issues'])} issues found"
                )
                self.logger.error(f"Debug info: {validation_result['debug_info']}")

            return validation_result

        except Exception as e:
            self.logger.error(f"Voice mapping validation failed: {e}")
            return {
                "valid": False,
                "missing_mappings": [],
                "speaker_distribution": {},
                "voice_coverage": {},
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "recommendations": ["Check script segments and voice profiles format"],
                "total_segments": 0,
                "validation_timestamp": time.time(),
                "debug_info": {"error": str(e)},
            }

    def validate_voice_balance(
        self, voice_segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ensure both hosts get roughly equal speaking time

        Args:
            voice_segments: List of voice segments with duration information

        Returns:
            Voice balance validation result
        """
        try:
            balance_result = {
                "balanced": True,
                "host_durations": {},
                "total_duration": 0.0,
                "balance_ratio": 1.0,
                "issues": [],
                "warnings": [],
                "recommendations": [],
            }

            if not voice_segments:
                balance_result["balanced"] = False
                balance_result["issues"].append(
                    "No voice segments provided for balance validation"
                )
                return balance_result

            # Calculate duration per host
            host_durations = {}
            total_duration = 0.0

            for segment in voice_segments:
                speaker = segment.get("speaker", "unknown")
                duration = segment.get("duration", 0.0)

                # Estimate duration if not provided (based on text length)
                if duration == 0.0:
                    text = segment.get("text", "")
                    # Rough estimate: 150 words per minute, 5 chars per word
                    estimated_duration = len(text) / (150 * 5 / 60)
                    duration = max(estimated_duration, 1.0)  # Minimum 1 second

                host_durations[speaker] = host_durations.get(speaker, 0.0) + duration
                total_duration += duration

            balance_result["host_durations"] = host_durations
            balance_result["total_duration"] = round(total_duration, 2)

            # Calculate balance ratio
            host_times = list(host_durations.values())
            if len(host_times) >= 2:
                max_time = max(host_times)
                min_time = min(host_times)
                if min_time > 0:
                    balance_result["balance_ratio"] = round(max_time / min_time, 2)
                else:
                    balance_result["balance_ratio"] = float("inf")
                    balance_result["issues"].append("One host has zero speaking time")
                    balance_result["balanced"] = False

            # Check balance thresholds
            if balance_result["balance_ratio"] > 2.0:
                balance_result["balanced"] = False
                balance_result["issues"].append(
                    f"Severe imbalance: ratio {balance_result['balance_ratio']}"
                )
            elif balance_result["balance_ratio"] > 1.5:
                balance_result["warnings"].append(
                    f"Moderate imbalance: ratio {balance_result['balance_ratio']}"
                )

            # Generate balance recommendations
            if not balance_result["balanced"]:
                balance_result["recommendations"].append(
                    "Consider redistributing dialogue more evenly between hosts"
                )
                balance_result["recommendations"].append(
                    "Review script generation parameters for better balance"
                )

            self.logger.debug(
                f"Voice balance validation: ratio {balance_result['balance_ratio']}, balanced: {balance_result['balanced']}"
            )
            return balance_result

        except Exception as e:
            self.logger.error(f"Voice balance validation failed: {e}")
            return {
                "balanced": False,
                "host_durations": {},
                "total_duration": 0.0,
                "balance_ratio": 0.0,
                "issues": [f"Balance validation error: {str(e)}"],
                "warnings": [],
                "recommendations": ["Check voice segments format and duration data"],
            }

    def validate_voice_availability(
        self, voice_assignments: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Validate that assigned voices are actually available

        Args:
            voice_assignments: Dict mapping speakers to voice IDs

        Returns:
            Voice availability validation result
        """
        try:
            # Import here to avoid circular dependencies
            from .chatterbox_service import chatterbox_service

            availability_result = {
                "all_available": True,
                "available_voices": [],
                "unavailable_voices": [],
                "voice_status": {},
                "issues": [],
                "warnings": [],
                "recommendations": [],
            }

            if not voice_assignments:
                availability_result["all_available"] = False
                availability_result["issues"].append("No voice assignments provided")
                return availability_result

            # Get available voices from chatterbox service
            try:
                available_voices = chatterbox_service.get_available_voices()
                # Fix: Use 'voice_id' field instead of 'id' field
                available_voice_ids = [
                    voice.get("voice_id", "") for voice in available_voices
                ]
                availability_result["available_voices"] = available_voice_ids
            except Exception as e:
                availability_result["warnings"].append(
                    f"Could not check voice availability: {e}"
                )
                # Continue with basic validation
                available_voice_ids = []

            # Check each assigned voice
            for speaker, voice_id in voice_assignments.items():
                if not voice_id:
                    availability_result["voice_status"][speaker] = {
                        "voice_id": None,
                        "available": False,
                        "status": "no_voice_assigned",
                    }
                    availability_result["issues"].append(
                        f"No voice assigned to {speaker}"
                    )
                    availability_result["all_available"] = False
                elif available_voice_ids and voice_id not in available_voice_ids:
                    availability_result["voice_status"][speaker] = {
                        "voice_id": voice_id,
                        "available": False,
                        "status": "voice_not_found",
                    }
                    availability_result["unavailable_voices"].append(voice_id)
                    availability_result["issues"].append(
                        f"Voice {voice_id} not available for {speaker}"
                    )
                    availability_result["all_available"] = False
                else:
                    availability_result["voice_status"][speaker] = {
                        "voice_id": voice_id,
                        "available": True,
                        "status": "available",
                    }

            # Generate recommendations for issues
            if availability_result["unavailable_voices"]:
                availability_result["recommendations"].append(
                    "Use available system voices or check voice ID spelling"
                )
                availability_result["recommendations"].append(
                    f"Available voices: {', '.join(available_voice_ids[:5])}..."
                )

            if not availability_result["all_available"]:
                availability_result["recommendations"].append(
                    "Validate voice assignments before generation"
                )

            self.logger.debug(
                f"Voice availability validation: {len(voice_assignments)} voices checked"
            )
            return availability_result

        except Exception as e:
            self.logger.error(f"Voice availability validation failed: {e}")
            return {
                "all_available": False,
                "available_voices": [],
                "unavailable_voices": [],
                "voice_status": {},
                "issues": [f"Availability validation error: {str(e)}"],
                "warnings": [],
                "recommendations": [
                    "Check voice assignment format and chatterbox service availability"
                ],
            }

    def comprehensive_validation(
        self,
        script_segments: List[Dict[str, Any]],
        voice_profiles: Dict[str, Any],
        user_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run comprehensive validation of all voice assignment aspects

        Args:
            script_segments: Script segments to validate
            voice_profiles: Available voice profiles
            user_inputs: User configuration for context

        Returns:
            Comprehensive validation result
        """
        try:
            comprehensive_result = {
                "overall_valid": True,
                "validation_timestamp": time.time(),
                "validations": {},
                "summary": {
                    "total_issues": 0,
                    "total_warnings": 0,
                    "critical_issues": [],
                    "recommendations": [],
                },
            }

            # Run speaker mapping validation
            mapping_validation = self.validate_speaker_mapping(
                script_segments, voice_profiles
            )
            comprehensive_result["validations"]["speaker_mapping"] = mapping_validation

            if not mapping_validation["valid"]:
                comprehensive_result["overall_valid"] = False

            # Run voice balance validation
            balance_validation = self.validate_voice_balance(script_segments)
            comprehensive_result["validations"]["voice_balance"] = balance_validation

            if not balance_validation["balanced"]:
                comprehensive_result["summary"]["recommendations"].append(
                    "Address voice balance issues"
                )

            # Extract voice assignments for availability check
            voice_assignments = {}
            for segment in script_segments:
                speaker = segment.get("speaker")
                voice_id = segment.get("voice_id")
                if speaker and voice_id:
                    voice_assignments[speaker] = voice_id

            # Run voice availability validation
            availability_validation = self.validate_voice_availability(
                voice_assignments
            )
            comprehensive_result["validations"]["voice_availability"] = (
                availability_validation
            )

            if not availability_validation["all_available"]:
                comprehensive_result["overall_valid"] = False

            # Aggregate summary statistics
            total_issues = 0
            total_warnings = 0
            critical_issues = []
            all_recommendations = []

            for validation_name, validation_result in comprehensive_result[
                "validations"
            ].items():
                issues = validation_result.get("issues", [])
                warnings = validation_result.get("warnings", [])
                recommendations = validation_result.get("recommendations", [])

                total_issues += len(issues)
                total_warnings += len(warnings)

                # Mark critical issues
                for issue in issues:
                    critical_issues.append(f"{validation_name}: {issue}")

                all_recommendations.extend(recommendations)

            comprehensive_result["summary"]["total_issues"] = total_issues
            comprehensive_result["summary"]["total_warnings"] = total_warnings
            comprehensive_result["summary"]["critical_issues"] = critical_issues
            comprehensive_result["summary"]["recommendations"] = list(
                set(all_recommendations)
            )  # Remove duplicates

            # Log comprehensive summary
            if comprehensive_result["overall_valid"]:
                self.logger.info(
                    f"Comprehensive voice validation PASSED: {total_warnings} warnings"
                )
            else:
                self.logger.error(
                    f"Comprehensive voice validation FAILED: {total_issues} issues, {total_warnings} warnings"
                )

            return comprehensive_result

        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")
            return {
                "overall_valid": False,
                "validation_timestamp": time.time(),
                "validations": {},
                "summary": {
                    "total_issues": 1,
                    "total_warnings": 0,
                    "critical_issues": [f"Validation system error: {str(e)}"],
                    "recommendations": [
                        "Check validation system and input data format"
                    ],
                },
            }

    def _validate_speaker_balance(
        self, speaker_distribution: Dict[str, Dict[str, Any]]
    ) -> List[str]:
        """
        Internal method to validate speaker balance

        Args:
            speaker_distribution: Distribution data with counts and percentages

        Returns:
            List of balance warning messages
        """
        warnings = []

        if len(speaker_distribution) < 2:
            warnings.append("Only one speaker found - expected at least 2 hosts")
            return warnings

        percentages = [data["percentage"] for data in speaker_distribution.values()]
        max_percentage = max(percentages)
        min_percentage = min(percentages)

        # Check for severe imbalance (one speaker dominates)
        if max_percentage > 80:
            warnings.append(
                f"Severe speaker imbalance: one speaker has {max_percentage}% of dialogue"
            )
        elif max_percentage > 70:
            warnings.append(
                f"Moderate speaker imbalance: max {max_percentage}%, min {min_percentage}%"
            )

        return warnings

    def _generate_mapping_recommendations(
        self,
        speaker_distribution: Dict[str, Dict[str, Any]],
        missing_mappings: List[str],
        voice_profiles: Dict[str, Any],
    ) -> List[str]:
        """
        Generate recommendations based on mapping validation results

        Args:
            speaker_distribution: Speaker distribution data
            missing_mappings: List of unmapped speakers
            voice_profiles: Available voice profiles

        Returns:
            List of recommendation messages
        """
        recommendations = []

        if missing_mappings:
            available_voices = list(voice_profiles.keys())
            recommendations.append(
                f"Map missing speakers to available voices: {', '.join(available_voices)}"
            )
            recommendations.append(
                "Ensure script uses consistent speaker identifiers (host_1, host_2)"
            )

        if len(speaker_distribution) > 2:
            recommendations.append(
                "Consider consolidating to 2 main hosts for better voice assignment"
            )

        if not speaker_distribution:
            recommendations.append(
                "Ensure script contains dialogue with speaker assignments"
            )

        return recommendations
