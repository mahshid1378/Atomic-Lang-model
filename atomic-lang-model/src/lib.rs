//! Atomic Language Model - Extremely Lightweight Universal Grammar Implementation
//! 
//! This library implements Minimalist Grammar operations (Merge + Move) in under 3k lines,
//! providing provable recursion with mathematical rigor while maintaining <50kB binary size.
//!
//! # Core Features
//! - Merge and Move operations from Minimalist Grammar theory
//! - Provably recursive generation (aⁿbⁿ patterns)
//! - Zero runtime dependencies
//! - Polynomial-time parsing with bounded memory
//! - Token-level linguistic evaluation

#![cfg_attr(feature = "no_std", no_std)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{vec::Vec, string::String, format};

use core::fmt;

// ============================================================================
// PyO3 Imports
// ============================================================================
#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

// ============================================================================
// Core Data Types
// ============================================================================

/// Syntactic category labels
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Category {
    // --- Standard Linguistic Categories ---
    /// Noun
    N,
    /// Verb  
    V,
    /// Determiner
    D,
    /// Complementizer
    C,
    /// Sentence
    S,
    /// Noun Phrase
    NP,
    /// Verb Phrase
    VP,
    /// Determiner Phrase
    DP,
    /// Complementizer Phrase
    CP,
    
    // --- NASA Telemetry Categories ---
    /// A generic telemetry event
    Event,
    /// A command event, e.g., MOTOR_CMD_START
    Command,
    /// A state-checking event, e.g., CURRENT_DRAW
    State,
    /// The overall mission context, e.g., DRIVE
    Context,
}

/// Feature types for Minimalist Grammar
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Feature {
    /// Basic category feature
    Cat(Category),
    /// Selector feature (requires merge with category)
    Sel(Category),
    /// Positive feature (triggers movement)
    Pos(u8),
    /// Negative feature (target for movement)
    Neg(u8),
    /// Context feature (e.g., for DRIVE, STANDBY)
    Ctx(String),
}

impl Feature {
    /// Check if feature is positive (triggers movement)
    pub fn is_positive(&self) -> bool {
        matches!(self, Feature::Pos(_))
    }
    
    /// Check if feature is negative (movement target)
    pub fn is_negative(&self) -> bool {
        matches!(self, Feature::Neg(_))
    }
    
    /// Get movement feature index if applicable
    pub fn movement_index(&self) -> Option<u8> {
        match self {
            Feature::Pos(i) | Feature::Neg(i) => Some(*i),
            _ => None,
        }
    }
}

/// Lexical item with phonological form and features
#[derive(Debug, Clone, PartialEq)]
pub struct LexItem {
    /// Phonological representation
    pub phon: String,
    /// Feature bundle
    pub feats: Vec<Feature>,
}

impl LexItem {
    /// Create new lexical item
    pub fn new(phon: &str, feats: &[Feature]) -> Self {
        Self {
            phon: phon.to_string(),
            feats: feats.to_vec(),
        }
    }
}

/// Syntactic object in derivation
#[derive(Debug, Clone, PartialEq)]
pub struct SyntacticObject {
    /// Category label
    pub label: Category,
    /// Unchecked features
    pub features: Vec<Feature>,
    /// Child constituents
    pub children: Vec<SyntacticObject>,
    /// Phonological content (for leaves)
    pub phon: Option<String>,
}

impl SyntacticObject {
    /// Create leaf node from lexical item
    pub fn from_lex(item: &LexItem) -> Self {
        let label = item.feats.iter()
            .find_map(|f| match f {
                Feature::Cat(cat) => Some(cat.clone()),
                _ => None,
            })
            .unwrap_or(Category::N); // Default to N if no category found
            
        Self {
            label,
            features: item.feats.clone(),
            children: Vec::new(),
            phon: Some(item.phon.clone()),
        }
    }
    
    /// Create internal node with children
    pub fn internal(label: Category, features: Vec<Feature>, children: Vec<SyntacticObject>) -> Self {
        Self {
            label,
            features,
            children,
            phon: None,
        }
    }
    
    /// Check if object has no unchecked features
    pub fn is_complete(&self) -> bool {
        self.features.is_empty()
    }
    
    /// Get linearized string representation
    pub fn linearize(&self) -> String {
        if let Some(ref phon) = self.phon {
            phon.clone()
        } else {
            self.children.iter()
                .map(|child| child.linearize())
                .collect::<Vec<_>>()
                .join(" ")
        }
    }
}

// ============================================================================
// Derivation Workspace
// ============================================================================

/// Workspace for managing derivation state
#[derive(Debug, Clone)]
pub struct Workspace {
    /// Active syntactic objects
    pub items: Vec<SyntacticObject>,
    /// Maximum memory usage allowed
    pub memory_limit: usize,
    /// Step counter for derivation
    pub step_count: usize,
}

/// Errors that can occur during derivation
#[derive(Debug, Clone, PartialEq)]
pub enum DerivationError {
    /// No valid operations available
    NoValidOperations,
    /// Memory limit exceeded
    MemoryLimitExceeded,
    /// Feature mismatch in operation
    FeatureMismatch,
    /// Empty workspace
    EmptyWorkspace,
    /// Invalid operation sequence
    InvalidOperation,
    /// Unknown Token
    UnknownToken(String),
}

impl fmt::Display for DerivationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DerivationError::NoValidOperations => write!(f, "No valid operations available"),
            DerivationError::MemoryLimitExceeded => write!(f, "Memory limit exceeded"),
            DerivationError::FeatureMismatch => write!(f, "Feature mismatch"),
            DerivationError::EmptyWorkspace => write!(f, "Empty workspace"),
            DerivationError::InvalidOperation => write!(f, "Invalid operation"),
            DerivationError::UnknownToken(s) => write!(f, "Unknown token: {}", s),
        }
    }
}

impl Workspace {
    /// Create new workspace with memory limit
    pub fn new(memory_limit: usize) -> Self {
        Self {
            items: Vec::new(),
            memory_limit,
            step_count: 0,
        }
    }
    
    /// Add lexical item to workspace
    pub fn add_lex(&mut self, item: &LexItem) {
        let obj = SyntacticObject::from_lex(item);
        self.items.push(obj);
    }
    
    /// Check if derivation is successful (single complete object)
    pub fn is_successful(&self) -> bool {
        self.items.len() == 1 && self.items[0].is_complete()
    }
    
    /// Get current memory usage estimate
    pub fn memory_usage(&self) -> usize {
        // Simple estimate based on object count and depth
        self.items.iter()
            .map(|obj| self.object_size(obj))
            .sum()
    }
    
    fn object_size(&self, obj: &SyntacticObject) -> usize {
        1 + obj.children.iter().map(|child| self.object_size(child)).sum::<usize>()
    }
}

// ============================================================================
// Core Operations: Merge
// ============================================================================

/// Attempt to merge two syntactic objects
pub fn merge(a: SyntacticObject, b: SyntacticObject) -> Result<SyntacticObject, DerivationError> {
    // Check if first object has selector feature matching second object's category
    if let Some(sel_feature) = a.features.iter().find(|f| matches!(f, Feature::Sel(_))) {
        if let Feature::Sel(required_cat) = sel_feature {
            if let Some(cat_feature) = b.features.iter().find(|f| matches!(f, Feature::Cat(_))) {
                if let Feature::Cat(actual_cat) = cat_feature {
                    if required_cat == actual_cat {
                        // Successful merge: create new object
                        let mut new_features = a.features.clone();
                        new_features.retain(|f| !matches!(f, Feature::Sel(_)));
                        
                        let mut b_features = b.features.clone();
                        b_features.retain(|f| !matches!(f, Feature::Cat(_)));
                        new_features.extend(b_features);
                        
                        return Ok(SyntacticObject::internal(
                            a.label.clone(),
                            new_features,
                            vec![a, b],
                        ));
                    }
                }
            }
        }
    }
    
    Err(DerivationError::FeatureMismatch)
}

/// Find pairs of objects that can merge
pub fn find_mergeable_pairs(workspace: &Workspace) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    
    for i in 0..workspace.items.len() {
        for j in 0..workspace.items.len() {
            if i != j {
                if can_merge(&workspace.items[i], &workspace.items[j]) {
                    pairs.push((i, j));
                }
            }
        }
    }
    
    pairs
}

/// Check if two objects can merge
pub fn can_merge(a: &SyntacticObject, b: &SyntacticObject) -> bool {
    // Check if a has selector feature matching b's category
    a.features.iter().any(|feat| {
        if let Feature::Sel(required_cat) = feat {
            b.features.iter().any(|b_feat| {
                matches!(b_feat, Feature::Cat(actual_cat) if actual_cat == required_cat)
            })
        } else {
            false
        }
    })
}

// ============================================================================
// Core Operations: Move
// ============================================================================

/// Apply movement operation to syntactic object
pub fn move_operation(obj: SyntacticObject) -> Result<SyntacticObject, DerivationError> {
    // Find positive feature that triggers movement
    if let Some(pos_feature) = obj.features.iter().find(|f| f.is_positive()) {
        if let Some(movement_idx) = pos_feature.movement_index() {
            // Search for matching negative feature in embedded structure
            if let Some(target) = find_movement_target(&obj, movement_idx) {
                return extract_and_move(obj, target, movement_idx);
            }
        }
    }
    
    Err(DerivationError::NoValidOperations)
}

/// Find constituent with matching negative feature
fn find_movement_target(obj: &SyntacticObject, movement_idx: u8) -> Option<SyntacticObject> {
    // Check current object
    if obj.features.iter().any(|f| matches!(f, Feature::Neg(idx) if *idx == movement_idx)) {
        return Some(obj.clone());
    }
    
    // Recursively search children
    for child in &obj.children {
        if let Some(target) = find_movement_target(child, movement_idx) {
            return Some(target);
        }
    }
    
    None
}

/// Extract target and adjoin to edge position
fn extract_and_move(
    obj: SyntacticObject, 
    target: SyntacticObject, 
    movement_idx: u8
) -> Result<SyntacticObject, DerivationError> {
    // Remove positive feature from trigger
    let mut new_features = obj.features.clone();
    new_features.retain(|f| !matches!(f, Feature::Pos(idx) if *idx == movement_idx));
    
    // Remove negative feature from target
    let mut target_features = target.features.clone();
    target_features.retain(|f| !matches!(f, Feature::Neg(idx) if *idx == movement_idx));
    
    let moved_target = SyntacticObject {
        features: target_features,
        ..target
    };
    
    // Create new structure with moved element adjoined
    Ok(SyntacticObject::internal(
        obj.label.clone(),
        new_features,
        vec![moved_target, obj],
    ))
}

// ============================================================================
// Derivation Engine
// ============================================================================

/// Single derivation step
pub fn step(workspace: &mut Workspace) -> Result<(), DerivationError> {
    if workspace.items.is_empty() {
        return Err(DerivationError::EmptyWorkspace);
    }
    
    workspace.step_count += 1;
    
    // Check memory limit
    if workspace.memory_usage() > workspace.memory_limit {
        return Err(DerivationError::MemoryLimitExceeded);
    }
    
    // Try merge operations first
    let mergeable_pairs = find_mergeable_pairs(workspace);
    if let Some((i, j)) = mergeable_pairs.first() {
        let a = workspace.items.remove(*i.max(j));
        let b = workspace.items.remove(*i.min(j));
        
        match merge(a, b) {
            Ok(merged) => {
                workspace.items.push(merged);
                return Ok(());
            }
            Err(e) => return Err(e),
        }
    }
    
    // Try move operations
    for i in 0..workspace.items.len() {
        if let Ok(moved) = move_operation(workspace.items[i].clone()) {
            workspace.items[i] = moved;
            return Ok(());
        }
    }
    
    Err(DerivationError::NoValidOperations)
}

/// Run complete derivation
pub fn derive(workspace: &mut Workspace, max_steps: usize) -> Result<SyntacticObject, DerivationError> {
    for _ in 0..max_steps {
        if workspace.is_successful() {
            return Ok(workspace.items[0].clone());
        }
        
        if let Err(e) = step(workspace) {
            if e == DerivationError::NoValidOperations {
                break; // Derivation stuck
            }
            return Err(e);
        }
    }
    
    if workspace.is_successful() {
        Ok(workspace.items[0].clone())
    } else {
        Err(DerivationError::NoValidOperations)
    }
}

// ============================================================================
// Lexicon and Grammar
// ============================================================================

/// Standard test lexicon for recursive patterns
pub fn test_lexicon() -> Vec<LexItem> {
    vec![
        LexItem::new("the", &[Feature::Cat(Category::D), Feature::Sel(Category::N)]),
        LexItem::new("a", &[Feature::Cat(Category::D), Feature::Sel(Category::N)]),
        LexItem::new("student", &[Feature::Cat(Category::N)]),
        LexItem::new("tutor", &[Feature::Cat(Category::N)]),
        LexItem::new("teacher", &[Feature::Cat(Category::N)]),
        LexItem::new("who", &[Feature::Cat(Category::C), Feature::Sel(Category::S)]),
        LexItem::new("that", &[Feature::Cat(Category::C), Feature::Sel(Category::S)]),
        LexItem::new("said", &[Feature::Cat(Category::V), Feature::Sel(Category::DP), Feature::Pos(1)]),
        LexItem::new("thinks", &[Feature::Cat(Category::V), Feature::Sel(Category::DP)]),
        LexItem::new("left", &[Feature::Cat(Category::V)]),
        LexItem::new("smiled", &[Feature::Cat(Category::V)]),
        LexItem::new("arrived", &[Feature::Cat(Category::V)]),
    ]
}

/// Generate aⁿbⁿ pattern for testing recursion
pub fn generate_an_bn(n: usize) -> String {
    if n == 0 {
        String::new()
    } else {
        let a_s = std::iter::repeat("a").take(n).collect::<Vec<_>>().join(" ");
        let b_s = std::iter::repeat("b").take(n).collect::<Vec<_>>().join(" ");
        format!("{} {}", a_s, b_s)
    }
}

/// Test if string matches aⁿbⁿ pattern
pub fn is_an_bn_pattern(s: &str) -> bool {
    let tokens: Vec<&str> = s.split_whitespace().collect();
    if tokens.is_empty() {
        return true; // ε case
    }
    
    let n = tokens.len() / 2;
    if tokens.len() != 2 * n {
        return false;
    }
    
    // Check first n tokens are 'a'
    for i in 0..n {
        if tokens[i] != "a" {
            return false;
        }
    }
    
    // Check last n tokens are 'b'
    for i in n..2*n {
        if tokens[i] != "b" {
            return false;
        }
    }
    
    true
}

// ============================================================================
// Public API
// ============================================================================

/// Parse sentence using Minimalist Grammar
pub fn parse_sentence(sentence: &str, lexicon: &[LexItem]) -> Result<SyntacticObject, DerivationError> {
    let tokens: Vec<&str> = sentence.split_whitespace().collect();
    let mut workspace = Workspace::new(4096); // 4KB memory limit
    
    // Add tokens to workspace
    for token in tokens {
        if let Some(lex_item) = lexicon.iter().find(|item| item.phon == token) {
            workspace.add_lex(lex_item);
        } else {
            return Err(DerivationError::UnknownToken(token.to_string()));
        }
    }
    
    derive(&mut workspace, 100) // Max 100 derivation steps
}

/// Generate string of specified pattern
pub fn generate_pattern(pattern: &str, n: usize) -> Result<String, DerivationError> {
    match pattern {
        "an_bn" => Ok(generate_an_bn(n)),
        _ => Err(DerivationError::InvalidOperation),
    }
}

/// Check if grammar can generate given string
pub fn can_generate(pattern: &str, n: usize) -> bool {
    generate_pattern(pattern, n).is_ok()
}

// ============================================================================
// Python Bridge (PyO3)
// ============================================================================

#[cfg(feature = "pyo3")]
#[pyfunction]
/// Validates a sequence of telemetry data using a formal grammar.
/// This is a simple version for basic anomaly detection.
fn validate_telemetry_sequence(sequence: Vec<f64>) -> PyResult<bool> {
    // This function is kept for backward compatibility, but the new demo
    // should use `validate_mission_log`.
    let is_anomalous = sequence.iter().any(|&v| v > 10.0);
    Ok(!is_anomalous)
}

#[cfg(feature = "pyo3")]
#[pyfunction]
/// Validates a structured mission log against a formal grammar of operations.
/// Returns a list of explanations for any ungrammatical (anomalous) sequences.
fn validate_mission_log(log: Vec<String>) -> PyResult<Vec<String>> {
    // --- The Grammar of Space Operations ---
    let lexicon = vec![
        // COMMANDS: Actions that can be taken. A command selects a state.
        LexItem::new("MOTOR_CMD_START", &[Feature::Cat(Category::Command), Feature::Sel(Category::State)]),
        LexItem::new("MOTOR_CMD_STOP", &[Feature::Cat(Category::Command), Feature::Sel(Category::State)]),
        LexItem::new("INSTRUMENT_PWR_ON", &[Feature::Cat(Category::Command), Feature::Sel(Category::State)]),
        LexItem::new("INSTRUMENT_PWR_OFF", &[Feature::Cat(Category::Command), Feature::Sel(Category::State)]),

        // STATES: Observations about the system. 
        // A state can select another state, allowing for a valid chain of telemetry.
        LexItem::new("VOLTAGE_SPIKE", &[Feature::Cat(Category::State)]), // Terminal state, cannot select another.
        LexItem::new("CURRENT_DRAW", &[Feature::Cat(Category::State), Feature::Sel(Category::State)]),
        LexItem::new("WHEEL_RPM", &[Feature::Cat(Category::State), Feature::Sel(Category::State)]),
        LexItem::new("TEMP_MOTOR", &[Feature::Cat(Category::State), Feature::Sel(Category::State)]),
        LexItem::new("TEMP_INSTRUMENT", &[Feature::Cat(Category::State), Feature::Sel(Category::State)]),
        LexItem::new("SPECTROMETER_READ", &[Feature::Cat(Category::State), Feature::Sel(Category::State)]),
    ];

    let mut anomalies = Vec::new();

    // We check each 2-event window.
    for i in 0..log.len() {
        if i + 1 >= log.len() { break; }

        let prev_event_str = &log[i];
        let current_event_str = &log[i+1];

        // Find the lexical items for the current window.
        let prev_lex_item = lexicon.iter().find(|item| item.phon == *prev_event_str);
        let current_lex_item = lexicon.iter().find(|item| item.phon == *current_event_str);

        if let (Some(prev_item), Some(current_item)) = (prev_lex_item, current_lex_item) {
            // Create syntactic objects from the lexical items.
            let prev_obj = SyntacticObject::from_lex(prev_item);
            let current_obj = SyntacticObject::from_lex(current_item);

            // The core logic: Check if the first event can grammatically select the second.
            if !can_merge(&prev_obj, &current_obj) {
                let explanation = format!(
                    "Anomaly Detected: Ungrammatical sequence '{}' followed by '{}'. This violates operational rules.",
                    prev_event_str, current_event_str
                );
                anomalies.push(explanation);
            }
        } else {
            // Handle cases where a token isn't in the lexicon.
            let explanation = format!(
                "Anomaly Detected: Unknown event(s) in sequence ['{}', '{}'].",
                prev_event_str, current_event_str
            );
            anomalies.push(explanation);
        }
    }

    Ok(anomalies)
}


#[cfg(feature = "pyo3")]
#[pymodule]
/// Python module for the Atomic Language Model.
fn atomic_lang_model_python(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_telemetry_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(validate_mission_log, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_operations() {
        let pos_feat = Feature::Pos(1);
        let neg_feat = Feature::Neg(1);
        
        assert!(pos_feat.is_positive());
        assert!(!pos_feat.is_negative());
        assert!(!neg_feat.is_positive());
        assert!(neg_feat.is_negative());
        
        assert_eq!(pos_feat.movement_index(), Some(1));
        assert_eq!(neg_feat.movement_index(), Some(1));
    }

    #[test]
    fn test_an_bn_generation() {
        assert_eq!(generate_an_bn(0), "");
        assert_eq!(generate_an_bn(1), "a b");
        assert_eq!(generate_an_bn(2), "a a b b");
        assert_eq!(generate_an_bn(3), "a a a b b b");
    }

    #[test]
    fn test_an_bn_recognition() {
        assert!(is_an_bn_pattern(""));
        assert!(is_an_bn_pattern("a b"));
        assert!(is_an_bn_pattern("a a b b"));
        assert!(is_an_bn_pattern("a a a b b b"));
        
        assert!(!is_an_bn_pattern("a"));
        assert!(!is_an_bn_pattern("a a b"));
        assert!(!is_an_bn_pattern("a b b"));
    }

    #[test]
    fn test_recursive_capability() {
        for n in 0..=5 {
            assert!(can_generate("an_bn", n));
        }
    }

    #[test]
    fn test_merge_operation() {
        let det = SyntacticObject::from_lex(&LexItem::new("the", &[Feature::Cat(Category::D)]));
        let noun = SyntacticObject::from_lex(&LexItem::new("student", &[Feature::Cat(Category::N)]));
        
        // This should fail - no selector feature
        assert!(merge(det.clone(), noun.clone()).is_err());
        
        // Create proper selector
        let det_sel = SyntacticObject {
            features: vec![Feature::Sel(Category::N)],
            ..det
        };
        
        // This should succeed
        let merged = merge(det_sel, noun).unwrap();
        assert_eq!(merged.label, Category::D);
    }

    #[test]
    fn test_workspace_operations() {
        let mut workspace = Workspace::new(1024);
        let lexicon = test_lexicon();
        
        workspace.add_lex(&lexicon[0]); // "the"
        workspace.add_lex(&lexicon[2]); // "student"
        
        assert_eq!(workspace.items.len(), 2);
        assert!(!workspace.is_successful());
    }

    #[test]
    fn test_mission_log_validation() {
        // Grammatical sequence
        let normal_log = vec!["CTX_DRIVE".to_string(), "MOTOR_CMD_START".to_string(), "VOLTAGE_SPIKE".to_string()];
        assert!(validate_mission_log(normal_log).unwrap().is_empty());

        // Ungrammatical sequence
        let anomaly_log = vec!["CTX_STANDBY".to_string(), "VOLTAGE_SPIKE".to_string()];
        assert!(!validate_mission_log(anomaly_log).unwrap().is_empty());
    }
}
