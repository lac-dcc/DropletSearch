// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensor_shape.proto

#include "tensor_shape.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/stubs/port.h>
#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// This is a temporary google only hack
#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
#include "third_party/protobuf/version.h"
#endif
// @@protoc_insertion_point(includes)
namespace tensorflow {
class TensorShapeProto_DimDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<TensorShapeProto_Dim>
      _instance;
} _TensorShapeProto_Dim_default_instance_;
class TensorShapeProtoDefaultTypeInternal {
 public:
  ::google::protobuf::internal::ExplicitlyConstructed<TensorShapeProto>
      _instance;
} _TensorShapeProto_default_instance_;
}  // namespace tensorflow
namespace protobuf_tensor_5fshape_2eproto {
void InitDefaultsTensorShapeProto_DimImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  ::google::protobuf::internal::InitProtobufDefaultsForceUnique();
#else
  ::google::protobuf::internal::InitProtobufDefaults();
#endif  // GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  {
    void* ptr = &::tensorflow::_TensorShapeProto_Dim_default_instance_;
    new (ptr) ::tensorflow::TensorShapeProto_Dim();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::TensorShapeProto_Dim::InitAsDefaultInstance();
}

void InitDefaultsTensorShapeProto_Dim() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &InitDefaultsTensorShapeProto_DimImpl);
}

void InitDefaultsTensorShapeProtoImpl() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

#ifdef GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  ::google::protobuf::internal::InitProtobufDefaultsForceUnique();
#else
  ::google::protobuf::internal::InitProtobufDefaults();
#endif  // GOOGLE_PROTOBUF_ENFORCE_UNIQUENESS
  protobuf_tensor_5fshape_2eproto::InitDefaultsTensorShapeProto_Dim();
  {
    void* ptr = &::tensorflow::_TensorShapeProto_default_instance_;
    new (ptr) ::tensorflow::TensorShapeProto();
    ::google::protobuf::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::TensorShapeProto::InitAsDefaultInstance();
}

void InitDefaultsTensorShapeProto() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &InitDefaultsTensorShapeProtoImpl);
}

::google::protobuf::Metadata file_level_metadata[2];

const ::google::protobuf::uint32 TableStruct::offsets[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::TensorShapeProto_Dim, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::TensorShapeProto_Dim, size_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::TensorShapeProto_Dim, name_),
  ~0u,  // no _has_bits_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::TensorShapeProto, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::TensorShapeProto, dim_),
  GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(::tensorflow::TensorShapeProto, unknown_rank_),
};
static const ::google::protobuf::internal::MigrationSchema schemas[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::TensorShapeProto_Dim)},
  { 7, -1, sizeof(::tensorflow::TensorShapeProto)},
};

static ::google::protobuf::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::google::protobuf::Message*>(&::tensorflow::_TensorShapeProto_Dim_default_instance_),
  reinterpret_cast<const ::google::protobuf::Message*>(&::tensorflow::_TensorShapeProto_default_instance_),
};

void protobuf_AssignDescriptors() {
  AddDescriptors();
  ::google::protobuf::MessageFactory* factory = NULL;
  AssignDescriptors(
      "tensor_shape.proto", schemas, file_default_instances, TableStruct::offsets, factory,
      file_level_metadata, NULL, NULL);
}

void protobuf_AssignDescriptorsOnce() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &protobuf_AssignDescriptors);
}

void protobuf_RegisterTypes(const ::std::string&) GOOGLE_PROTOBUF_ATTRIBUTE_COLD;
void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::internal::RegisterAllTypes(file_level_metadata, 2);
}

void AddDescriptorsImpl() {
  InitDefaults();
  static const char descriptor[] GOOGLE_PROTOBUF_ATTRIBUTE_SECTION_VARIABLE(protodesc_cold) = {
      "\n\022tensor_shape.proto\022\ntensorflow\"z\n\020Tens"
      "orShapeProto\022-\n\003dim\030\002 \003(\0132 .tensorflow.T"
      "ensorShapeProto.Dim\022\024\n\014unknown_rank\030\003 \001("
      "\010\032!\n\003Dim\022\014\n\004size\030\001 \001(\003\022\014\n\004name\030\002 \001(\tBq\n\030"
      "org.tensorflow.frameworkB\021TensorShapePro"
      "tosP\001Z=github.com/tensorflow/tensorflow/"
      "tensorflow/go/core/framework\370\001\001b\006proto3"
  };
  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
      descriptor, 279);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "tensor_shape.proto", &protobuf_RegisterTypes);
}

void AddDescriptors() {
  static GOOGLE_PROTOBUF_DECLARE_ONCE(once);
  ::google::protobuf::GoogleOnceInit(&once, &AddDescriptorsImpl);
}
// Force AddDescriptors() to be called at dynamic initialization time.
struct StaticDescriptorInitializer {
  StaticDescriptorInitializer() {
    AddDescriptors();
  }
} static_descriptor_initializer;
}  // namespace protobuf_tensor_5fshape_2eproto
namespace tensorflow {

// ===================================================================

void TensorShapeProto_Dim::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TensorShapeProto_Dim::kSizeFieldNumber;
const int TensorShapeProto_Dim::kNameFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

TensorShapeProto_Dim::TensorShapeProto_Dim()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    ::protobuf_tensor_5fshape_2eproto::InitDefaultsTensorShapeProto_Dim();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.TensorShapeProto.Dim)
}
TensorShapeProto_Dim::TensorShapeProto_Dim(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena) {
  ::protobuf_tensor_5fshape_2eproto::InitDefaultsTensorShapeProto_Dim();
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.TensorShapeProto.Dim)
}
TensorShapeProto_Dim::TensorShapeProto_Dim(const TensorShapeProto_Dim& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  if (from.name().size() > 0) {
    name_.Set(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), from.name(),
      GetArenaNoVirtual());
  }
  size_ = from.size_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.TensorShapeProto.Dim)
}

void TensorShapeProto_Dim::SharedCtor() {
  name_.UnsafeSetDefault(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
  size_ = GOOGLE_LONGLONG(0);
  _cached_size_ = 0;
}

TensorShapeProto_Dim::~TensorShapeProto_Dim() {
  // @@protoc_insertion_point(destructor:tensorflow.TensorShapeProto.Dim)
  SharedDtor();
}

void TensorShapeProto_Dim::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
  name_.DestroyNoArena(&::google::protobuf::internal::GetEmptyStringAlreadyInited());
}

void TensorShapeProto_Dim::ArenaDtor(void* object) {
  TensorShapeProto_Dim* _this = reinterpret_cast< TensorShapeProto_Dim* >(object);
  (void)_this;
}
void TensorShapeProto_Dim::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void TensorShapeProto_Dim::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* TensorShapeProto_Dim::descriptor() {
  ::protobuf_tensor_5fshape_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensor_5fshape_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const TensorShapeProto_Dim& TensorShapeProto_Dim::default_instance() {
  ::protobuf_tensor_5fshape_2eproto::InitDefaultsTensorShapeProto_Dim();
  return *internal_default_instance();
}

TensorShapeProto_Dim* TensorShapeProto_Dim::New(::google::protobuf::Arena* arena) const {
  return ::google::protobuf::Arena::CreateMessage<TensorShapeProto_Dim>(arena);
}

void TensorShapeProto_Dim::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.TensorShapeProto.Dim)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  name_.ClearToEmpty(&::google::protobuf::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  size_ = GOOGLE_LONGLONG(0);
  _internal_metadata_.Clear();
}

bool TensorShapeProto_Dim::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.TensorShapeProto.Dim)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // int64 size = 1;
      case 1: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(8u /* 8 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int64, ::google::protobuf::internal::WireFormatLite::TYPE_INT64>(
                 input, &size_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string name = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          DO_(::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
            this->name().data(), static_cast<int>(this->name().length()),
            ::google::protobuf::internal::WireFormatLite::PARSE,
            "tensorflow.TensorShapeProto.Dim.name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.TensorShapeProto.Dim)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.TensorShapeProto.Dim)
  return false;
#undef DO_
}

void TensorShapeProto_Dim::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.TensorShapeProto.Dim)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 size = 1;
  if (this->size() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteInt64(1, this->size(), output);
  }

  // string name = 2;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.TensorShapeProto.Dim.name");
    ::google::protobuf::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->name(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.TensorShapeProto.Dim)
}

::google::protobuf::uint8* TensorShapeProto_Dim::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.TensorShapeProto.Dim)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // int64 size = 1;
  if (this->size() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt64ToArray(1, this->size(), target);
  }

  // string name = 2;
  if (this->name().size() > 0) {
    ::google::protobuf::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::google::protobuf::internal::WireFormatLite::SERIALIZE,
      "tensorflow.TensorShapeProto.Dim.name");
    target =
      ::google::protobuf::internal::WireFormatLite::WriteStringToArray(
        2, this->name(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.TensorShapeProto.Dim)
  return target;
}

size_t TensorShapeProto_Dim::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.TensorShapeProto.Dim)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // string name = 2;
  if (this->name().size() > 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::StringSize(
        this->name());
  }

  // int64 size = 1;
  if (this->size() != 0) {
    total_size += 1 +
      ::google::protobuf::internal::WireFormatLite::Int64Size(
        this->size());
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void TensorShapeProto_Dim::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.TensorShapeProto.Dim)
  GOOGLE_DCHECK_NE(&from, this);
  const TensorShapeProto_Dim* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const TensorShapeProto_Dim>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.TensorShapeProto.Dim)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.TensorShapeProto.Dim)
    MergeFrom(*source);
  }
}

void TensorShapeProto_Dim::MergeFrom(const TensorShapeProto_Dim& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.TensorShapeProto.Dim)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.name().size() > 0) {
    set_name(from.name());
  }
  if (from.size() != 0) {
    set_size(from.size());
  }
}

void TensorShapeProto_Dim::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.TensorShapeProto.Dim)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TensorShapeProto_Dim::CopyFrom(const TensorShapeProto_Dim& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.TensorShapeProto.Dim)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TensorShapeProto_Dim::IsInitialized() const {
  return true;
}

void TensorShapeProto_Dim::Swap(TensorShapeProto_Dim* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    TensorShapeProto_Dim* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void TensorShapeProto_Dim::UnsafeArenaSwap(TensorShapeProto_Dim* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void TensorShapeProto_Dim::InternalSwap(TensorShapeProto_Dim* other) {
  using std::swap;
  name_.Swap(&other->name_);
  swap(size_, other->size_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata TensorShapeProto_Dim::GetMetadata() const {
  protobuf_tensor_5fshape_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensor_5fshape_2eproto::file_level_metadata[kIndexInFileMessages];
}


// ===================================================================

void TensorShapeProto::InitAsDefaultInstance() {
}
#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int TensorShapeProto::kDimFieldNumber;
const int TensorShapeProto::kUnknownRankFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

TensorShapeProto::TensorShapeProto()
  : ::google::protobuf::Message(), _internal_metadata_(NULL) {
  if (GOOGLE_PREDICT_TRUE(this != internal_default_instance())) {
    ::protobuf_tensor_5fshape_2eproto::InitDefaultsTensorShapeProto();
  }
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.TensorShapeProto)
}
TensorShapeProto::TensorShapeProto(::google::protobuf::Arena* arena)
  : ::google::protobuf::Message(),
  _internal_metadata_(arena),
  dim_(arena) {
  ::protobuf_tensor_5fshape_2eproto::InitDefaultsTensorShapeProto();
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.TensorShapeProto)
}
TensorShapeProto::TensorShapeProto(const TensorShapeProto& from)
  : ::google::protobuf::Message(),
      _internal_metadata_(NULL),
      dim_(from.dim_),
      _cached_size_(0) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  unknown_rank_ = from.unknown_rank_;
  // @@protoc_insertion_point(copy_constructor:tensorflow.TensorShapeProto)
}

void TensorShapeProto::SharedCtor() {
  unknown_rank_ = false;
  _cached_size_ = 0;
}

TensorShapeProto::~TensorShapeProto() {
  // @@protoc_insertion_point(destructor:tensorflow.TensorShapeProto)
  SharedDtor();
}

void TensorShapeProto::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == NULL);
}

void TensorShapeProto::ArenaDtor(void* object) {
  TensorShapeProto* _this = reinterpret_cast< TensorShapeProto* >(object);
  (void)_this;
}
void TensorShapeProto::RegisterArenaDtor(::google::protobuf::Arena* arena) {
}
void TensorShapeProto::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* TensorShapeProto::descriptor() {
  ::protobuf_tensor_5fshape_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensor_5fshape_2eproto::file_level_metadata[kIndexInFileMessages].descriptor;
}

const TensorShapeProto& TensorShapeProto::default_instance() {
  ::protobuf_tensor_5fshape_2eproto::InitDefaultsTensorShapeProto();
  return *internal_default_instance();
}

TensorShapeProto* TensorShapeProto::New(::google::protobuf::Arena* arena) const {
  return ::google::protobuf::Arena::CreateMessage<TensorShapeProto>(arena);
}

void TensorShapeProto::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.TensorShapeProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  dim_.Clear();
  unknown_rank_ = false;
  _internal_metadata_.Clear();
}

bool TensorShapeProto::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!GOOGLE_PREDICT_TRUE(EXPRESSION)) goto failure
  ::google::protobuf::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.TensorShapeProto)
  for (;;) {
    ::std::pair< ::google::protobuf::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .tensorflow.TensorShapeProto.Dim dim = 2;
      case 2: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(18u /* 18 & 0xFF */)) {
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessage(input, add_dim()));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bool unknown_rank = 3;
      case 3: {
        if (static_cast< ::google::protobuf::uint8>(tag) ==
            static_cast< ::google::protobuf::uint8>(24u /* 24 & 0xFF */)) {

          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   bool, ::google::protobuf::internal::WireFormatLite::TYPE_BOOL>(
                 input, &unknown_rank_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.TensorShapeProto)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.TensorShapeProto)
  return false;
#undef DO_
}

void TensorShapeProto::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.TensorShapeProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.TensorShapeProto.Dim dim = 2;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->dim_size()); i < n; i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      2, this->dim(static_cast<int>(i)), output);
  }

  // bool unknown_rank = 3;
  if (this->unknown_rank() != 0) {
    ::google::protobuf::internal::WireFormatLite::WriteBool(3, this->unknown_rank(), output);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.TensorShapeProto)
}

::google::protobuf::uint8* TensorShapeProto::InternalSerializeWithCachedSizesToArray(
    bool deterministic, ::google::protobuf::uint8* target) const {
  (void)deterministic; // Unused
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.TensorShapeProto)
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.TensorShapeProto.Dim dim = 2;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->dim_size()); i < n; i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      InternalWriteMessageToArray(
        2, this->dim(static_cast<int>(i)), deterministic, target);
  }

  // bool unknown_rank = 3;
  if (this->unknown_rank() != 0) {
    target = ::google::protobuf::internal::WireFormatLite::WriteBoolToArray(3, this->unknown_rank(), target);
  }

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.TensorShapeProto)
  return target;
}

size_t TensorShapeProto::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.TensorShapeProto)
  size_t total_size = 0;

  if ((_internal_metadata_.have_unknown_fields() &&  ::google::protobuf::internal::GetProto3PreserveUnknownsDefault())) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        (::google::protobuf::internal::GetProto3PreserveUnknownsDefault()   ? _internal_metadata_.unknown_fields()   : _internal_metadata_.default_instance()));
  }
  // repeated .tensorflow.TensorShapeProto.Dim dim = 2;
  {
    unsigned int count = static_cast<unsigned int>(this->dim_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::google::protobuf::internal::WireFormatLite::MessageSize(
          this->dim(static_cast<int>(i)));
    }
  }

  // bool unknown_rank = 3;
  if (this->unknown_rank() != 0) {
    total_size += 1 + 1;
  }

  int cached_size = ::google::protobuf::internal::ToCachedSize(total_size);
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = cached_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void TensorShapeProto::MergeFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.TensorShapeProto)
  GOOGLE_DCHECK_NE(&from, this);
  const TensorShapeProto* source =
      ::google::protobuf::internal::DynamicCastToGenerated<const TensorShapeProto>(
          &from);
  if (source == NULL) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.TensorShapeProto)
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.TensorShapeProto)
    MergeFrom(*source);
  }
}

void TensorShapeProto::MergeFrom(const TensorShapeProto& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.TensorShapeProto)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::google::protobuf::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  dim_.MergeFrom(from.dim_);
  if (from.unknown_rank() != 0) {
    set_unknown_rank(from.unknown_rank());
  }
}

void TensorShapeProto::CopyFrom(const ::google::protobuf::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.TensorShapeProto)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void TensorShapeProto::CopyFrom(const TensorShapeProto& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.TensorShapeProto)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool TensorShapeProto::IsInitialized() const {
  return true;
}

void TensorShapeProto::Swap(TensorShapeProto* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    TensorShapeProto* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == NULL) {
      delete temp;
    }
  }
}
void TensorShapeProto::UnsafeArenaSwap(TensorShapeProto* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void TensorShapeProto::InternalSwap(TensorShapeProto* other) {
  using std::swap;
  dim_.InternalSwap(&other->dim_);
  swap(unknown_rank_, other->unknown_rank_);
  _internal_metadata_.Swap(&other->_internal_metadata_);
  swap(_cached_size_, other->_cached_size_);
}

::google::protobuf::Metadata TensorShapeProto::GetMetadata() const {
  protobuf_tensor_5fshape_2eproto::protobuf_AssignDescriptorsOnce();
  return ::protobuf_tensor_5fshape_2eproto::file_level_metadata[kIndexInFileMessages];
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow

// @@protoc_insertion_point(global_scope)
