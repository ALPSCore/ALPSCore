/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2011 by Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/**
 * @file fwrapper_impl.C
 * @brief Fortranから呼ばれるラッパ関数群の定義
 */

#include <alps/parameter/parameters.h>

#include "fortran_wrapper.h"
#include "fwrapper_impl.h"

#include <string.h>
#include <string>
#include <valarray>

extern "C"
{
  void alps_get_parameter_(void* data, char* name, int* type, int* caller, int ldata, int lname);
  void alps_parameter_defined_(int* res, char* name, int* caller, int lname);
  void alps_dump_(void* data, int* count,  int* type, int* caller, int lchar);
  void alps_restore_(void* data, int* count,  int* type, int* caller, int lchar);
  void alps_init_observable_(int* count, int* type, char* name, int* caller, int lname);
  void alps_accumulate_observable_(void* data, int* count, int* type, char* name, int* caller, int lname);
}

namespace
{
  const char FORTRAN_BLANK = ' '; //!< Fortranで余白領域にセットされる文字(スペース)

  /**
   * @brief データの型を表すスイッチ。
   * @note alps_fortran.h と合わせる必要があるため、編集時は要注意。
   *
   */
  enum ALPS_TYPE_CODE
    {
      ALPS_CHAR = 0,
      ALPS_INT,
      ALPS_LONG,
      ALPS_REAL,
      ALPS_DOUBLE_PRECISION,
    };

  /**
   * @brief Fortran文字列からstd::stringを生成する。
   *
   * Fortran文字列には終端文字がないため、
   * 文字列内で最も後ろにある「スペースでない文字」の直後に終端を挿入し、
   * std::stringを生成する。
   *
   * @param[in] str Fortran文字列
   * @param[in] len strの長さ
   * @return 変換結果
   */
  std::string getString(const char* str, int len)
  {
    int i, ll=len;
    char* tmp = new char[len+1];
    memset(tmp, 0x00, len+1);

    for(i=len-1; i>=0; i--)
      {
	if( str[i] != FORTRAN_BLANK )
	  {
	    ll = i+1;
	    break;
	  }
      }
    strncpy(tmp,str,ll);
    std::string retval = tmp;
    delete [] tmp;
    return retval;
  }

  /**
   * @brief Fortranの文字列配列をstd::vector<std::string> に変換する。
   *
   * 文字列の変換ロジック自体は getString と同じ。
   * @param[in] str Fortran文字列配列
   * @param[in] len 文字列1つの長さ
   * @param[in] count 文字列の数
   * @return 変換結果
   */
  std::vector< std::string > getStrings(const char* str, int len, int count)
  {
    std::vector<std::string> retval;
    for(int i = 0; i < count; ++i)
      {
	retval.push_back(getString(str+i*len, len));
      }
    return retval;
  }

  /**
   * @brief C++文字列(std::string)をFortran文字列に変換する。
   * 
   * len > src.size() の場合、dstの残り要素にはスペースを入れる。
   * 終端文字は挿入されないので、dstを文字列として使用することはできないので注意。
   * @param[out] dst 変換結果の格納先
   * @param[in] len dstの長さ
   * @param[in] src 変換元
   * @retval true 変換成功
   * @retval false 変換失敗(src.size() > len)
   */
  bool setString(char* dst, int len, const std::string& src)
  {
    if(src.size() > len)
      return false;

    memset(dst, FORTRAN_BLANK, len);
    memcpy(dst, src.c_str(), src.size());

    return true;
  }

  /**
   * @brief 指定されたパラメータをalps::Parameterから取得し、dataにセットする。
   * @param[out] data パラメータの格納先
   * @param[in] params パラメータセット
   * @param[in] name パラメータ名
   * @param[in] lname nameの長さ
   * @param[in] ldata dataの長さ。T=charの場合のみ使用される。
   */
  template <typename T> void getParameter(void* data, const alps::Parameters*& params, const char* name, int lname, int ldata = 0)
  {
    T* retval = reinterpret_cast<T*>(data);
    *retval = static_cast<T>(alps::evaluate(getString(name, lname).c_str(), *params));
  }

  /**
   * @brief 指定されたパラメータをalps::Parameterから取得し、dataにセットする。
   * @param[out] data パラメータの格納先
   * @param[in] params パラメータセット
   * @param[in] name パラメータ名
   * @param[in] lname nameの長さ
   * @param[in] ldata dataの長さ。T=charの場合のみ使用される。
   */
  template<> void getParameter<char>(void* data, const alps::Parameters*& params, const char* name, int lname, int ldata)
  {
    char* str = reinterpret_cast<char*>(data);
    if(!setString(str, ldata, (*params)[getString(name, lname)]))
      boost::throw_exception(std::runtime_error("alps_get_parameter :: buffer is too small."));
  }

  /**
   * @brief データをalps::ODumpに書き出す
   * @param[in] data 書き出すパラメータ
   * @param[in] count dataの要素数
   * @param[out] odump 出力先
   * @param[in] len dataの長さ。T=charの場合のみ使用される。
   */
  template <typename T> void dump(const void* data, int count, alps::ODump& odump, int len = 0)
  {
    const T* values = reinterpret_cast<const T*>(data);
    for(int i = 0; i < count; ++i)
      {
	odump << values[i];
      }
  }

  /**
   * @brief データをalps::ODumpに書き出す
   * @param[in] data 書き出すパラメータ
   * @param[in] count dataの要素数
   * @param[out] odump 出力先
   * @param[in] len dataの長さ。T=charの場合のみ使用される。
   */
  template <> void dump<char>(const void* data, int count, alps::ODump& odump, int len)
  {
    std::vector<std::string> values = getStrings(reinterpret_cast<const char*>(data), len, count);
    for(int i = 0; i < count; ++i)
      {
	odump << values[i];
      }
  }

  /**
   * @brief alps::IDumpからデータを取り出す
   * @param[out] data 取り出したデータの格納先
   * @param[in] count dataの要素数
   * @param[in] idump 入力元
   * @param[in] len dataの長さ。T=charの場合のみ使用される。
   */
  template <typename T> void restore(void* data, int count, alps::IDump& idump, int len = 0)
  {
    T* values = reinterpret_cast<T*>(data);
    for(int i = 0; i < count; ++i)
      {
	idump >> values[i];
      }
  }

  /**
   * @brief alps::IDumpからデータを取り出す
   * @param[out] data 取り出したデータの格納先
   * @param[in] count dataの要素数
   * @param[in] idump 入力元
   * @param[in] len dataの長さ。T=charの場合のみ使用される。
   */
  template <> void restore<char>(void* data, int count, alps::IDump& idump, int len)
  {
    char* values = reinterpret_cast<char*>(data);
    std::string tmp;
    for(int i = 0; i < count; ++i)
      {
	idump >> tmp;
	if(!setString(values + i*len, len, tmp))
	  boost::throw_exception(std::runtime_error("alps_resotre :: buffer is too small."));
      }
  }

  /**
   * @brief 指定された名前のObservableにデータを追加する
   * @param[in] data 追加されるデータ
   * @param[in] count dataの長さ
   * @param[out] obs データの追加先
   * @param[in] name Observableの名前
   */
  template <typename T> void accumulate(void* data, int count, alps::ObservableSet& obs, const std::string& name)
  {
    std::valarray<T> array(reinterpret_cast<T*>(data), count);

    if(count == 1)
	obs[name] << array[0];
    else
	obs[name] << array;
  }

  /**
   * @brief 指定された名前のObservableにデータを追加する
   * @param[in] data 追加されるデータ
   * @param[in] count dataの長さ
   * @param[out] obs データの追加先
   * @param[in] name Observableの名前
   */
  template <> void accumulate<float>(void* data, int count, alps::ObservableSet& obs, const std::string& name)
  {
    std::valarray<float> array(count);
    float* values = reinterpret_cast<float*>(data);
    for(int i = 0; i < count; ++i)
      array[i] = static_cast<double>(values[i]);

    if(count == 1)
      obs[name] << array[0];
    else
      obs[name] << array;
  }
}

/**
 * @brief alps::Parameters内に指定されたパラメータが定義されているかどうかを返す
 *
 * @param[out] res 結果格納先(1:定義されている / 0:定義されていない)
 * @param[in] name 取得するパラメータの名前
 * @param[in] caller alps::fortran_wrapper インスタンスへのポインタ
 * @param[in] lname nameの長さ
 */
void alps_parameter_defined_(int* res, char* name, int* caller, int lname)
{
  alps::fortran_wrapper::alps_fortran_ptr ptr(caller);
  const alps::Parameters* params = ptr.m_pointer->parameters();

  *res = params->defined(getString(name, lname));
}

/**
 * @brief alps::Parametersからデータを取得する
 *
 * @param[out] data 取得したデータの格納先
 * @param[in] name 取得するパラメータの名前
 * @param[in] type パラメータの型
 * @param[in] caller alps::fortran_wrapper インスタンスへのポインタ
 * @param[in] len1 nameの長さ。ただし、type == ALPS_CHARの場合はdataの長さ。
 * @param[in] len2 nameの長さ。type == ALPS_CHARの場合のみ有効(それ以外の場合、値は不定)。
 */
void alps_get_parameter_(void* data, char* name, int* type, int* caller, int len1, int len2)
{
  alps::fortran_wrapper::alps_fortran_ptr ptr(caller);
  const alps::Parameters* params = ptr.m_pointer->parameters();

  switch(*type)
    {
    case ALPS_CHAR:
      getParameter<char>(data, params, name, len2, len1);
      break;
    case ALPS_INT:
      getParameter<int>(data, params, name, len1);
      break;
    case ALPS_LONG:
      getParameter<long>(data, params, name, len1);
      break;
    case ALPS_REAL:
      getParameter<float>(data, params, name, len1);
      break;
    case ALPS_DOUBLE_PRECISION:
      getParameter<double>(data, params, name, len1);
      break;
    default:
      boost::throw_exception(std::runtime_error("alps_get_parameter : an invalid type is specified."));
      break;
    }
}

/**
 * @brief alps::ODumpを通してデータをダンプする
 * 
 * @param[in] data ダンプするデータ
 * @param[in] count dataの要素数
 * @param[in] type dataの型
 * @param[in] caller alps::fortran_wrapper インスタンスへのポインタ
 * @param[in] lchar dataの長さ。ただし、type == ALPS_CHARの場合のみ有効。
 */
void alps_dump_(void* data, int* count,  int* type, int* caller, int lchar)
{
  alps::fortran_wrapper::alps_fortran_ptr ptr(caller);
  switch(*type)
    {
    case ALPS_CHAR:
      dump<char>(data, *count, ptr.m_pointer->odump(), lchar);
      break;
    case ALPS_INT:
      dump<int>(data, *count, ptr.m_pointer->odump());
      break;
    case ALPS_LONG:
      dump<long>(data, *count, ptr.m_pointer->odump());
      break;
    case ALPS_REAL:
      dump<float>(data, *count, ptr.m_pointer->odump());
      break;
    case ALPS_DOUBLE_PRECISION:
      dump<double>(data, *count, ptr.m_pointer->odump());
      break;
    default:
      boost::throw_exception(std::runtime_error("alps_dump : an invalid type is specified."));
      break;
    }
}

/**
 * @brief alps::IDumpを通してダンプデータをリストアする
 * 
 * @param[out] data リストアデータの格納先
 * @param[in] count dataの要素数
 * @param[in] type dataの型
 * @param[in] caller alps::fortran_wrapper インスタンスへのポインタ
 * @param[in] lchar dataの長さ。ただし、type == ALPS_CHARの場合のみ有効。
 */
void alps_restore_(void* data, int* count,  int* type, int* caller, int lchar)
{
  alps::fortran_wrapper::alps_fortran_ptr ptr(caller);
  switch(*type)
    {
    case ALPS_CHAR:
      restore<char>(data, *count, ptr.m_pointer->idump(), lchar);
      break;
    case ALPS_INT:
      restore<int>(data, *count, ptr.m_pointer->idump());
      break;
    case ALPS_LONG:
      restore<long>(data, *count, ptr.m_pointer->idump());
      break;
    case ALPS_REAL:
      restore<float>(data, *count, ptr.m_pointer->idump());
      break;
    case ALPS_DOUBLE_PRECISION:
      restore<double>(data, *count, ptr.m_pointer->idump());
      break;
    default:
      boost::throw_exception(std::runtime_error("alps_restore : an invalid type is specified."));
      break;
    }
}

/**
 * @brief alps::ObservableSetにObservableを追加する
 * @param[in] count Observableに渡す値の要素数
 * @param[in] type  Observableに渡す値の型
 * @param[in] name  Observableの名前
 * @param[in] caller alps::fortran_wrapper インスタンスへのポインタ
 * @param[in] lname nameの長さ
 *
 * count > 1 のとき、[Int|Real]VectorObservableが追加される。
 * count == 1のとき、[Int|Real]Observableが追加される。
 */
void alps_init_observable_(int* count, int* type, char* name, int* caller, int lname)
{
  alps::fortran_wrapper::alps_fortran_ptr ptr(caller);
  alps::ObservableSet& obs = ptr.m_pointer->observables();
  std::string str = getString(name, lname);

  switch(*type)
    {
    case ALPS_INT:
      if(*count == 1)
	obs << alps::IntObservable(str);
      else
	obs << alps::IntVectorObservable(str);
      break;
    case ALPS_REAL:
    case ALPS_DOUBLE_PRECISION:
      if(*count == 1)
	obs << alps::RealObservable(str);
      else
	obs << alps::RealVectorObservable(str);
      break;
    default:
      boost::throw_exception(std::runtime_error("alps_init_observable : an invalid type is specified."));
      break;
    }
}

/**
 * @brief alps::Observableにデータを加算する
 * @param[in] data 加算する値
 * @param[in] count dataの要素数
 * @param[in] type  dataの型
 * @param[in] name  Observableの名前
 * @param[in] caller alps::fortran_wrapper インスタンスへのポインタ
 * @param[in] lname nameの長さ
 */
void alps_accumulate_observable_(void* data, int* count, int* type, char* name, int* caller, int lname)
{
  alps::fortran_wrapper::alps_fortran_ptr ptr(caller);
  alps::ObservableSet& obs = ptr.m_pointer->observables();
  std::string str = getString(name, lname);

  switch(*type)
    {
    case ALPS_INT:
      accumulate<int>(data, *count, obs, str);
      break;
    case ALPS_REAL:
      accumulate<float>(data, *count, obs, str);
      break;
    case ALPS_DOUBLE_PRECISION:
      accumulate<double>(data, *count, obs, str);
      break;
    default:
      boost::throw_exception(std::runtime_error("alps_accumulate_observable : an invalid type is specified."));
      break;
    }

}
